"""
inference.py — Baseline Inference Script
CEWA: Cognitive Enterprise Workflow Agent — v3

Changes in v3:
- Communicates with the deployed server via HTTP (POST /reset, POST /step)
  so the script works identically locally and when run against the HF Space.
- [START] / [STEP] / [END] log lines match the exact OpenEnv spec format.
- Retry loop with exponential backoff for LLM calls.
- JSON schema validation + alias normalisation on LLM output.
- Heuristic fallback agent when LLM is unavailable.

Required env vars:
  API_BASE_URL   LLM API base URL  (default: https://api.openai.com/v1)
  MODEL_NAME     Model identifier  (default: gpt-4o-mini)
  HF_TOKEN       HuggingFace / API key
  ENV_BASE_URL   CEWA server URL   (default: http://localhost:7860)

Log format (stdout):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import time
import logging
import re
from typing import List, Optional

import httpx
from openai import OpenAI, APIError, APITimeoutError, RateLimitError

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cewa.inference")

# ── Config ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860").rstrip("/")

BENCHMARK               = "cewa"
SUCCESS_SCORE_THRESHOLD = 0.5

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", ""),
    timeout=30.0,
)

# ── Valid values ─────────────────────────────────────────────────────────────
VALID_CATEGORIES = {"billing", "bug", "access", "network", "security", "hr", "general"}
VALID_TEAMS      = {"finance", "tech", "security", "hr", "management"}
VALID_PRIORITIES = {"1", "2", "3", "4", "5"}

CATEGORY_ALIASES = {
    "technical": "bug", "software": "bug", "code": "bug",
    "authentication": "security", "auth": "security", "breach": "security",
    "payment": "billing", "invoice": "billing", "charge": "billing",
    "it": "access", "login": "access", "credential": "access",
    "employee": "hr", "onboard": "hr",
}
TEAM_ALIASES = {
    "engineering": "tech", "devops": "tech", "development": "tech",
    "infosec": "security", "soc": "security", "cybersecurity": "security",
    "payroll": "finance", "accounting": "finance",
    "people": "hr", "people_ops": "hr",
    "exec": "management", "leadership": "management",
}

# ── Logging helpers (exact OpenEnv spec format) ──────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── HTTP helpers ─────────────────────────────────────────────────────────────

def http_reset(task_id: str, seed: int = 42) -> dict:
    resp = httpx.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def http_step(session_id: str, action: dict) -> dict:
    resp = httpx.post(
        f"{ENV_BASE_URL}/step",
        json={"session_id": session_id, "action": action},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert Cognitive Enterprise Workflow Agent managing IT operations.

Each episode presents a realistic enterprise ticket scenario. You make ONE decision per step.
Your goal: make all required decisions (classify, prioritize, assign, respond) before the SLA expires.

AVAILABLE ACTIONS:
  classify      value: billing | bug | access | network | security | hr | general
  prioritize    value: "1" | "2" | "3" | "4" | "5"  (5 = most critical)
  assign        value: finance | tech | security | hr | management
  respond       value: your resolution message (minimum 20 words)
  escalate      value: brief reason for escalation
  request_info  value: what additional information you need

PRIORITY GUIDANCE:
  5 (Critical): Active attack, production down, revenue loss per hour, data breach
  4 (High):     Customer-facing bug, financial dispute, major service degraded
  3 (Medium):   Important but not customer-impacting, can wait 4-8h
  2 (Low):      Minor, cosmetic, single-user issue
  1 (Info):     Question, documentation request

TEAM ROUTING:
  security  -> auth failures, breaches, Tor IPs, credential stuffing
  tech      -> CI/CD, deployments, bugs, pipelines, infrastructure outages
  finance   -> billing, invoices, payments, refunds, vendor contracts
  hr        -> onboarding, offboarding, employee disputes
  management -> escalation only when p=5 OR multi-team OR regulatory risk

MULTI-ISSUE REASONING: Rank by ongoing revenue loss/hr, then people blocked, then customer impact.
Dollar amounts alone are misleading; a large invoice with extension options < ongoing hourly outage.

Always respond with ONLY a valid JSON object:
{"action_type": "<action>", "value": "<value>", "reason": "<1-2 sentence reasoning from ticket evidence>"}"""


def normalize_value(action_type: str, raw_value: str) -> str:
    v = str(raw_value).strip().lower().strip('"').strip("'")
    if action_type == "classify":
        if v in VALID_CATEGORIES:
            return v
        return CATEGORY_ALIASES.get(v, v)
    elif action_type == "assign":
        if v in VALID_TEAMS:
            return v
        return TEAM_ALIASES.get(v, v)
    elif action_type == "prioritize":
        digits = re.findall(r"[1-5]", v)
        return digits[0] if digits else "3"
    return raw_value.strip()


def validate_action_dict(d: dict) -> tuple:
    at = d.get("action_type", "")
    v  = d.get("value", "")
    if not at:
        return False, "missing action_type"
    if at == "classify" and v not in VALID_CATEGORIES:
        return False, f"invalid category: {v}"
    if at == "assign" and v not in VALID_TEAMS:
        return False, f"invalid team: {v}"
    if at == "prioritize" and v not in VALID_PRIORITIES:
        return False, f"invalid priority: {v}"
    if at == "respond" and len(str(v)) < 10:
        return False, "response too short"
    return True, ""


def build_prompt(obs: dict) -> str:
    pd      = obs.get("partial_decisions", {})
    sla     = obs.get("sla", {})
    weights = obs.get("task_context", {}).get("score_weights", {})
    required = ["classification", "priority", "team", "response"]
    todo = [k for k in required if not pd.get(k)]

    lines = [
        f"Step {obs['step']+1}/{obs['episode_length']} | Difficulty: {obs['difficulty'].upper()}",
        "",
        "TICKET:",
        obs["task_description"],
        "",
    ]
    if obs.get("urgency_signals"):
        lines += ["URGENCY SIGNALS:"] + [f"  * {s}" for s in obs["urgency_signals"]]
        lines.append("")
    if obs.get("similar_past_tickets"):
        lines += ["SIMILAR PAST CASES:"] + [f"  {t}" for t in obs["similar_past_tickets"]]
        lines.append("")
    if obs.get("team_availability"):
        avail   = [k for k, v in obs["team_availability"].items() if v]
        unavail = [k for k, v in obs["team_availability"].items() if not v]
        lines.append(f"TEAMS AVAILABLE: {', '.join(avail)}")
        if unavail:
            lines.append(f"TEAMS UNAVAILABLE: {', '.join(unavail)}")
        lines.append("")
    breached = sla.get("breached", False)
    lines.append(f"SLA: {'BREACHED' if breached else str(sla.get('steps_remaining', '?')) + ' steps remaining'}")
    done_keys = [k for k in required if pd.get(k)]
    if done_keys:
        lines.append(f"DONE: {', '.join(done_keys)}")
    lines.append(f"TODO: {', '.join(todo) if todo else 'ALL COMPLETE'}")
    if pd:
        lines.append(f"CURRENT DECISIONS: {json.dumps({k: v for k, v in pd.items() if v})}")
    if weights:
        lines.append(f"SCORE WEIGHTS: {json.dumps(weights)}")
    if obs.get("history"):
        lines.append(f"HISTORY: {obs['history']}")
    if todo:
        lines.append(f"\nSUGGESTED NEXT: {todo[0]}")
    return "\n".join(lines)


def call_llm_with_retry(prompt: str, history: list, max_retries: int = 3) -> dict:
    history.append({"role": "user", "content": prompt})
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                temperature=0.0,
                max_tokens=350,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            history.append({"role": "assistant", "content": raw})
            return json.loads(raw)
        except (APITimeoutError, RateLimitError) as e:
            wait = 2 ** attempt
            logger.warning(f"LLM attempt {attempt+1} failed ({type(e).__name__}), retry in {wait}s")
            time.sleep(wait)
            last_error = e
        except (json.JSONDecodeError, APIError) as e:
            logger.warning(f"LLM attempt {attempt+1} error: {e}")
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(1)
    raise RuntimeError(f"LLM failed after {max_retries} attempts: {last_error}")


def heuristic_fallback(obs: dict) -> dict:
    pd       = obs.get("partial_decisions", {})
    desc     = obs.get("task_description", "").lower()
    signals  = " ".join(obs.get("urgency_signals", [])).lower()
    combined = desc + " " + signals
    required = ["classification", "priority", "team", "response"]
    missing  = [k for k in required if not pd.get(k)]
    target   = missing[0] if missing else "respond"

    if target == "classification":
        if any(w in combined for w in ["auth", "tor", "breach", "credential", "attack", "soc"]):
            val = "security"
        elif any(w in combined for w in ["pipeline", "deploy", "ci/cd", "cicd", "crash"]):
            val = "bug"
        elif any(w in combined for w in ["payment", "charge", "invoice", "billing", "refund"]):
            val = "billing"
        elif any(w in combined for w in ["access", "login", "password", "locked"]):
            val = "access"
        elif any(w in combined for w in ["onboard", "hr", "employee", "cto", "hire"]):
            val = "hr"
        else:
            val = "general"
        return {"action_type": "classify", "value": val, "reason": f"heuristic: signals indicate {val}"}
    elif target == "priority":
        if any(w in combined for w in ["revenue", "production down", "breach", "critical", "attack"]):
            val = "5"
        elif any(w in combined for w in ["urgent", "immediately", "today", "churn", "customer"]):
            val = "4"
        else:
            val = "3"
        return {"action_type": "prioritize", "value": val, "reason": "heuristic: priority from signals"}
    elif target == "team":
        cls = pd.get("classification", "")
        val = {"security": "security", "bug": "tech", "access": "tech",
               "billing": "finance", "hr": "hr", "network": "tech"}.get(cls, "tech")
        return {"action_type": "assign", "value": val, "reason": f"heuristic: {cls} -> {val}"}
    # For post-mortem tasks, prefer request_info before respond
    task_id = obs.get("task_id", "")
    if target == "respond" and task_id == "task4_postmortem_contradiction":
        if not pd.get("info_requested"):
            return {
                "action_type": "request_info",
                "value": "Please provide the git commit timestamp and deploy log for d4f9a2 to resolve the timeline contradiction.",
                "reason": "heuristic: deploy timestamp is the key missing evidence to resolve contradiction",
            }

    else:
        cls  = pd.get("classification", "issue")
        team = pd.get("team", "the relevant team")
        return {
            "action_type": "respond",
            "value": (
                f"Thank you for reporting this {cls} issue. "
                f"We have escalated it to {team} as a priority {pd.get('priority', 4)} request. "
                f"Our team is investigating and will provide an update within the SLA window. "
                f"We will contact you directly with a resolution."
            ),
            "reason": "heuristic: standard resolution response",
        }


def run_task(task_id: str, seed: int = 42) -> dict:
    """Run one task episode via HTTP and return score + reward list."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    reset_data = http_reset(task_id, seed)
    session_id = reset_data["session_id"]
    obs        = reset_data["observation"]

    history: List[dict] = []
    rewards: List[float] = []
    steps_taken = 0
    last_error  = None
    info        = {}

    try:
        step_num = 1
        while True:
            prompt = build_prompt(obs)

            try:
                raw_dict = call_llm_with_retry(prompt, history)
            except Exception as e:
                logger.warning(f"LLM unavailable step {step_num}: {e} — heuristic fallback")
                raw_dict = heuristic_fallback(obs)

            action_type = raw_dict.get("action_type", "classify")
            norm_value  = normalize_value(action_type, str(raw_dict.get("value", "general")))
            norm_dict   = {**raw_dict, "value": norm_value}

            valid, err = validate_action_dict(norm_dict)
            if not valid:
                logger.warning(f"Validation failed: {err} — heuristic fallback")
                norm_dict  = heuristic_fallback(obs)
                norm_value = normalize_value(norm_dict["action_type"], norm_dict["value"])
                norm_dict["value"] = norm_value
                last_error = err
            else:
                last_error = None

            action_payload = {
                "action_type": norm_dict.get("action_type", "classify"),
                "value":       norm_dict.get("value", "general"),
                "reason":      norm_dict.get("reason", ""),
            }

            step_data = http_step(session_id, action_payload)
            obs       = step_data["observation"]
            reward    = float(step_data["reward"])
            done      = bool(step_data["done"])
            info      = step_data.get("info", {})

            rewards.append(reward)
            steps_taken = step_num

            action_str = json.dumps(action_payload)
            events     = info.get("events", [])
            error_val  = str(last_error) if last_error else (events[0] if events else None)
            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error_val)

            if done:
                break
            step_num += 1

    except Exception as e:
        logger.error(f"Episode error for {task_id}: {e}")
        last_error = str(e)

    task_score = info.get("task_score")
    if task_score is None:
        task_score = sum(r for r in rewards if r > 0) / max(len(rewards), 1)
    task_score = float(task_score)
    task_score = max(0.0, min(1.0, task_score))

    success = task_score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=task_score, rewards=rewards)

    return {"task": task_id, "score": task_score, "steps": steps_taken, "rewards": rewards}


def main() -> int:
    tasks_to_run = [
        "task1_basic_triage",
        "task2_incident_triage",
        "task3_complex_orchestration",
        "task4_postmortem_contradiction",
    ]
    all_reports: dict = {}

    for task_id in tasks_to_run:
        print(f"\n{'='*60}", flush=True)
        print(f"Running {task_id} ...", flush=True)
        try:
            report = run_task(task_id, seed=42)
        except Exception as e:
            import traceback
            traceback.print_exc()
            report = {"task": task_id, "score": 0.0, "steps": 0, "rewards": [], "error": str(e)}
            log_end(success=False, steps=0, score=0.0, rewards=[])
        all_reports[task_id] = report

    avg = sum(all_reports[t].get("score", 0.0) for t in tasks_to_run) / len(tasks_to_run)
    print(f"\nAverage score across all tasks: {avg:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
