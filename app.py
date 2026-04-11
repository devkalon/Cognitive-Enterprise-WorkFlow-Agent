"""
app.py — FastAPI server for HuggingFace Space
CEWA: Cognitive Enterprise Workflow Agent — OpenEnv

Endpoints: /health, /reset, /step, /state, /tasks, /grade
"""
import os
import sys
import uuid
import importlib
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

from env.environment import CEWAEnv
from env.state import Action

app = FastAPI(
    title="CEWA — Cognitive Enterprise Workflow Agent",
    description="OpenEnv environment for enterprise IT ticket triage and workflow management.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_sessions: Dict[str, CEWAEnv] = {}


# ── Request models ─────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    session_id: str
    action: Dict[str, Any]


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/")
@app.get("/health")
async def health():
    """Health check — must return 200 for HF Space automated validation."""
    return {"status": "ok", "env": "cognitive-enterprise-workflow-agent", "version": "1.0.0"}


@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None):
    """Reset environment and return initial observation."""

    # SAFE defaults
    task_id = None
    seed = 42

    if req is not None:
        if req.task_id is not None:
            task_id = req.task_id
        if req.seed is not None:
            seed = req.seed

    env = CEWAEnv(task_id=task_id, seed=seed)
    obs = env.reset(seed=seed)

    sid = str(uuid.uuid4())[:12]
    _sessions[sid] = env

    return {
        "session_id": sid,
        "observation": obs.model_dump(),
        "task_id": getattr(obs, "task_id", task_id),
    }


@app.post("/step")
async def step(req: StepRequest):
    """Execute one action, return (observation, reward, done, info)."""
    env = _sessions.get(req.session_id)
    if not env:
        raise HTTPException(404, f"Session {req.session_id} not found. Call /reset first.")
    try:
        action = Action(**req.action)
    except Exception as e:
        raise HTTPException(422, f"Invalid action: {e}")
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    if done:
        _sessions.pop(req.session_id, None)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info.model_dump(),
    }


@app.get("/state")
async def state(session_id: str):
    """Return current state without stepping."""
    env = _sessions.get(session_id)
    if not env:
        raise HTTPException(404, f"Session {session_id} not found.")
    return {"observation": env.state().model_dump()}


@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    from tasks import load_tasks
    tasks = load_tasks()
    return {"tasks": [
        {
            "id": t["id"],
            "name": t["name"],
            "difficulty": t["difficulty"],
            "episode_length": t["episode_length"],
            "description": t["description"][:200] + "...",
        }
        for t in tasks
    ]}


@app.post("/grade")
async def grade(task_id: str, seed: int = 42):
    """Run a full episode with rule-based baseline and return graded score."""
    from tasks import get_task
    from graders import get_grader

    try:
        task = get_task(task_id)
    except ValueError:
        raise HTTPException(404, f"Unknown task: {task_id}")

    grader = get_grader(task_id)
    env = CEWAEnv(task_id=task_id, seed=seed)
    obs = env.reset(seed=seed)
    done = False

    while not done:
        action = _rule_agent(obs)
        next_obs, _, done, info = env.step(action)
        grader.on_step(obs, action, next_obs, info)
        obs = next_obs

    return grader.report(obs, task)


def _rule_agent(obs) -> Action:
    """Rule-based agent from your baseline_agent.py — used for /grade endpoint."""
    pd = obs.partial_decisions
    desc = obs.task_description.lower()

    if "classification" not in pd:
        val = "billing" if "payment" in desc or "charge" in desc or "invoice" in desc \
            else "security" if "auth" in desc or "access" in desc \
            else "bug" if "crash" in desc or "pipeline" in desc \
            else "general"
        return Action(action_type="classify", value=val, reason="rule-based classification")

    if "priority" not in pd:
        val = "5" if "critical" in desc or "urgent" in desc or "immediately" in desc else "4"
        return Action(action_type="prioritize", value=val, reason="rule-based priority")

    if "team" not in pd:
        cls = pd.get("classification", "")
        val = "finance" if cls == "billing" else \
              "security" if cls == "security" else \
              "hr" if cls == "hr" else "tech"
        return Action(action_type="assign", value=val, reason="rule-based assignment")

    if "response" not in pd:
        cls = pd.get("classification", "issue")
        return Action(
            action_type="respond",
            value=f"Thank you for contacting us. We have classified your {cls} request "
                  f"and assigned it to the relevant team. We will resolve this within our SLA window.",
            reason="rule-based response"
        )

    return Action(action_type="classify", value="general", reason="all done")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
