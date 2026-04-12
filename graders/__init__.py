"""
CEWA Graders — v2
3 deterministic graders, one per task. Score range: 0.0–1.0.

Upgrades over v1:
- Use response_must_include AND response_must_not_include from task spec
- Reasoning quality scored by length + specificity
- Partial credit for priority ±1
- Distractor detection: extra penalty if agent chose a distractor category on T3
- SLA discipline: multiplier on final score
- Full details dict for debugging and judge review
"""
from __future__ import annotations
from typing import Dict, Any
from env.state import Observation, Action, StepInfo

def safe_score(score: float) -> float:
    EPS = 1e-6
    if score <= 0:
        return EPS
    if score >= 1:
        return 1 - EPS
    return score


def _response_score(response: str, task: Dict[str, Any]) -> float:
    """Score response 0.0–1.0 using must_include and must_not_include."""
    rv        = (response or "").lower()
    must_inc  = task.get("response_must_include", [])
    must_not  = task.get("response_must_not_include", [])
    matches   = sum(1 for w in must_inc if w in rv)
    forbidden = sum(1 for w in must_not if w in rv)
    score     = min(1.0, matches / max(len(must_inc), 1))
    score    -= forbidden * 0.20
    return max(0.0, round(score, 4))


def _priority_score(given, true_priority: int) -> float:
    """Priority score with partial credit."""
    try:
        diff = abs(int(given) - true_priority)
    except (ValueError, TypeError):
        return 0.0
    if diff == 0: return 1.0
    if diff == 1: return 0.50
    if diff == 2: return 0.15
    return 0.0


def _reasoning_score(reasons_given: int, difficulty: str) -> float:
    """Normalize reasoning quality by task difficulty."""
    needed = {"easy": 1, "medium": 2, "hard": 3}[difficulty]
    return min(1.0, reasons_given / needed)


# ── Grader 1 — Easy ────────────────────────────────────────────────────────

class Task1Grader:
    """
    task1_basic_triage (Easy)
    classification 30% | priority 20% | team 25% | response 25%
    No escalation required. SLA multiplier: ×0.85 if breached.
    """
    def __init__(self):
        self.steps         = 0
        self.sla_breached  = False
        self.reasons_given = 0

    def on_step(self, obs: Observation, action: Action, next_obs: Observation, info: StepInfo):
        self.steps += 1
        if next_obs.sla.breached:
            self.sla_breached = True
        if action.reason and len(action.reason.strip()) >= 8:
            self.reasons_given += 1

    def score(self, final_obs: Observation, task: Dict[str, Any]) -> float:
        pd = final_obs.partial_decisions
        w  = task["task_context"]["score_weights"]

        c  = w["classification"]   * (1.0 if pd.get("classification") == task["true_category"] else 0.0)
        p  = w["priority"]         * _priority_score(pd.get("priority"), task["true_priority"])
        t  = w["team_assignment"]  * (1.0 if pd.get("team") == task["true_team"] else 0.0)
        r  = w["response_quality"] * _response_score(pd.get("response", ""), task)

        total = c + p + t + r
        if self.sla_breached:
            total *= 0.85
            total = max(0.0, min(1.0, total))
            total = safe_score(total)
            return round(total, 4)

    def report(self, final_obs: Observation, task: Dict[str, Any]) -> dict:
        pd = final_obs.partial_decisions
        return {
            "task":       task["id"],
            "difficulty": "easy",
            "score":      self.score(final_obs, task),
            "details": {
                "classification":          pd.get("classification"),
                "expected_classification": task["true_category"],
                "classification_correct":  pd.get("classification") == task["true_category"],
                "priority_given":          pd.get("priority"),
                "expected_priority":       task["true_priority"],
                "priority_score":          _priority_score(pd.get("priority"), task["true_priority"]),
                "team_assigned":           pd.get("team"),
                "expected_team":           task["true_team"],
                "team_correct":            pd.get("team") == task["true_team"],
                "response_quality":        _response_score(pd.get("response", ""), task),
                "response_drafted":        bool(pd.get("response")),
                "sla_breached":            self.sla_breached,
                "steps_taken":             self.steps,
                "reasons_given":           self.reasons_given,
                "grader_notes":            task["task_context"].get("grader_notes", ""),
            }
        }


# ── Grader 2 — Medium ─────────────────────────────────────────────────────

class Task2Grader:
    """
    task2_incident_triage (Medium)
    classification 25% | priority 20% | team 25% | escalation 15% | response 15%
    Extra penalty if agent assigned to "tech" instead of "security".
    SLA multiplier: ×0.80 if breached (tighter than easy).
    """
    def __init__(self):
        self.steps         = 0
        self.escalated     = False
        self.sla_breached  = False
        self.reasons_given = 0
        self.assigned_tech = False   # distractor detection

    def on_step(self, obs: Observation, action: Action, next_obs: Observation, info: StepInfo):
        self.steps += 1
        if action.action_type == "escalate":
            self.escalated = True
        if action.action_type == "assign" and action.value == "tech":
            self.assigned_tech = True
        if next_obs.sla.breached:
            self.sla_breached = True
        if action.reason and len(action.reason.strip()) >= 8:
            self.reasons_given += 1

    def score(self, final_obs: Observation, task: Dict[str, Any]) -> float:
        pd = final_obs.partial_decisions
        w  = task["task_context"]["score_weights"]

        c = w["classification"]  * (1.0 if pd.get("classification") == task["true_category"] else 0.0)
        p = w["priority"]        * _priority_score(pd.get("priority"), task["true_priority"])

        # Team — extra penalty for choosing tech on security incident
        if pd.get("team") == task["true_team"]:
            t = w["team_assignment"] * 1.0
        elif self.assigned_tech:
            t = w["team_assignment"] * -0.30   # fell for distractor
        else:
            t = 0.0

        # Escalation (bidirectional)
        requires = task.get("requires_escalation", False)
        if requires and self.escalated:
            e = w["escalation_decision"] * 1.0
        elif not requires and not self.escalated:
            e = w["escalation_decision"] * 1.0
        elif requires and not self.escalated:
            e = w["escalation_decision"] * -0.50  # missed mandatory escalation
        else:
            e = w["escalation_decision"] * -0.25  # unnecessary escalation

        r = w["response_quality"] * _response_score(pd.get("response", ""), task)

        total = c + p + t + e + r
        if self.sla_breached:
            total *= 0.80
            total = max(0.0, min(1.0, total))
            total = safe_score(total)
            return round(total, 4)

    def report(self, final_obs: Observation, task: Dict[str, Any]) -> dict:
        pd = final_obs.partial_decisions
        return {
            "task":       task["id"],
            "difficulty": "medium",
            "score":      self.score(final_obs, task),
            "details": {
                "classification":          pd.get("classification"),
                "expected_classification": task["true_category"],
                "classification_correct":  pd.get("classification") == task["true_category"],
                "priority_given":          pd.get("priority"),
                "expected_priority":       task["true_priority"],
                "priority_score":          _priority_score(pd.get("priority"), task["true_priority"]),
                "team_assigned":           pd.get("team"),
                "expected_team":           task["true_team"],
                "team_correct":            pd.get("team") == task["true_team"],
                "fell_for_tech_distractor":  self.assigned_tech,
                "escalated":               self.escalated,
                "escalation_required":     task.get("requires_escalation"),
                "response_quality":        _response_score(pd.get("response", ""), task),
                "sla_breached":            self.sla_breached,
                "steps_taken":             self.steps,
                "reasons_given":           self.reasons_given,
                "grader_notes":            task["task_context"].get("grader_notes", ""),
            }
        }


# ── Grader 3 — Hard ───────────────────────────────────────────────────────

class Task3Grader:
    """
    task3_complex_orchestration (Hard)
    classification 20% | priority 20% | team 25% | escalation 15%
    response 10% | reasoning 10%
    Distractor penalty: chose billing or hr classification.
    SLA multiplier: ×0.75 if breached (hardest penalty).
    """
    def __init__(self):
        self.steps                 = 0
        self.escalated             = False
        self.sla_breached          = False
        self.reasons_given         = 0
        self.chose_billing         = False   # distractor A
        self.chose_hr              = False   # distractor B
        self.chose_finance_team    = False   # wrong team for distractor A
        self.chose_hr_team         = False   # wrong team for distractor B

    def on_step(self, obs: Observation, action: Action, next_obs: Observation, info: StepInfo):
        self.steps += 1
        if action.action_type == "escalate":
            self.escalated = True
        if action.action_type == "classify":
            if action.value == "billing":
                self.chose_billing = True
            elif action.value == "hr":
                self.chose_hr = True
        if action.action_type == "assign":
            if action.value == "finance":
                self.chose_finance_team = True
            elif action.value == "hr":
                self.chose_hr_team = True
        if next_obs.sla.breached:
            self.sla_breached = True
        if action.reason and len(action.reason.strip()) >= 15:
            self.reasons_given += 1

    def score(self, final_obs: Observation, task: Dict[str, Any]) -> float:
        pd = final_obs.partial_decisions
        w  = task["task_context"]["score_weights"]

        # Classification — distractor penalty
        if pd.get("classification") == task["true_category"]:
            c = w["classification"] * 1.0
        elif self.chose_billing or self.chose_hr:
            c = w["classification"] * -0.25  # fell for a distractor
        else:
            c = 0.0

        p = w["priority"] * _priority_score(pd.get("priority"), task["true_priority"])

        # Team — distractor penalty
        if pd.get("team") == task["true_team"]:
            t = w["team_assignment"] * 1.0
        elif self.chose_finance_team or self.chose_hr_team:
            t = w["team_assignment"] * -0.20  # chose distractor team
        else:
            t = 0.0

        # Escalation
        requires = task.get("requires_escalation", False)
        if requires and self.escalated:
            e = w["escalation_decision"] * 1.0
        elif not requires and not self.escalated:
            e = w["escalation_decision"] * 1.0
        elif requires and not self.escalated:
            e = w["escalation_decision"] * -0.60
        else:
            e = w["escalation_decision"] * -0.30

        r = w["response_quality"] * _response_score(pd.get("response", ""), task)

        rsn = w["reasoning_quality"] * _reasoning_score(self.reasons_given, "hard")

        total = c + p + t + e + r + rsn
        if self.sla_breached:
            total *= 0.75
            total = max(0.0, min(1.0, total))
            total = safe_score(total)
            return round(total, 4)

    def report(self, final_obs: Observation, task: Dict[str, Any]) -> dict:
        pd = final_obs.partial_decisions
        return {
            "task":       task["id"],
            "difficulty": "hard",
            "score":      self.score(final_obs, task),
            "details": {
                "classification":               pd.get("classification"),
                "expected_classification":      task["true_category"],
                "classification_correct":       pd.get("classification") == task["true_category"],
                "fell_for_billing_distractor":  self.chose_billing,
                "fell_for_hr_distractor":       self.chose_hr,
                "priority_given":               pd.get("priority"),
                "expected_priority":            task["true_priority"],
                "priority_score":               _priority_score(pd.get("priority"), task["true_priority"]),
                "team_assigned":                pd.get("team"),
                "expected_team":                task["true_team"],
                "team_correct":                 pd.get("team") == task["true_team"],
                "chose_finance_team":           self.chose_finance_team,
                "chose_hr_team":                self.chose_hr_team,
                "escalated":                    self.escalated,
                "escalation_required":          task.get("requires_escalation"),
                "response_quality":             _response_score(pd.get("response", ""), task),
                "reasoning_steps":              self.reasons_given,
                "reasoning_score":              _reasoning_score(self.reasons_given, "hard"),
                "sla_breached":                 self.sla_breached,
                "steps_taken":                  self.steps,
                "grader_notes":                 task["task_context"].get("grader_notes", ""),
            }
        }


# ── Registry ───────────────────────────────────────────────────────────────

_GRADER_MAP = {
    "task1_basic_triage":            Task1Grader,
    "task2_incident_triage":         Task2Grader,
    "task3_complex_orchestration":   Task3Grader,
    "task4_postmortem_contradiction": Task4Grader,
}


def get_grader(task_id: str):
    cls = _GRADER_MAP.get(task_id)
    if not cls:
        raise ValueError(f"No grader for task_id '{task_id}'. Valid: {list(_GRADER_MAP)}")
    return cls()


# ── Grader 4 — Expert ─────────────────────────────────────────────────────

class Task4Grader:
    """
    task4_postmortem_contradiction (Expert)
    Novel mechanic: info_request is SCORED (20%) — agent must ask for
    the git commit timestamp to resolve the deploy contradiction.
    Response must be blameless (no names), classification=bug, priority=3.
    Distractor: infra sounds credible but timeline is impossible.
    SLA multiplier: x0.80 if breached.
    """
    def __init__(self):
        self.steps            = 0
        self.sla_breached     = False
        self.reasons_given    = 0
        self.info_requested   = False
        self.info_request_hit = False   # did the request mention d4f9a/timestamp/git?
        self.chose_high_priority = False  # p=4 or p=5 when p=3 is correct

    def on_step(self, obs: Observation, action: Action, next_obs: Observation, info: StepInfo):
        self.steps += 1
        if next_obs.sla.breached:
            self.sla_breached = True
        if action.reason and len(action.reason.strip()) >= 15:
            self.reasons_given += 1
        if action.action_type == "request_info":
            self.info_requested = True
            val_lower = (action.value or "").lower()
            keywords = task_info_keywords = [
                "d4f9a", "commit", "git", "deploy", "timestamp", "log", "hash"
            ]
            if any(kw in val_lower for kw in keywords):
                self.info_request_hit = True
        if action.action_type == "prioritize":
            try:
                if int(action.value) >= 4:
                    self.chose_high_priority = True
            except (ValueError, TypeError):
                pass

    def score(self, final_obs: Observation, task: Dict[str, Any]) -> float:
        pd = final_obs.partial_decisions
        w  = task["task_context"]["score_weights"]

        # Classification
        c = w["classification"] * (1.0 if pd.get("classification") == task["true_category"] else 0.0)

        # Priority — p=3 is correct; p=4 gets partial; p=5 is wrong (outage resolved)
        p_given = pd.get("priority")
        try:
            diff = abs(int(p_given) - task["true_priority"])
        except (ValueError, TypeError):
            diff = 99
        if diff == 0:
            p = w["priority"] * 1.0
        elif diff == 1:
            p = w["priority"] * 0.40
        elif self.chose_high_priority:
            p = w["priority"] * -0.20   # confidently wrong: treating resolved incident as live
        else:
            p = 0.0

        # Team
        t = w["team_assignment"] * (1.0 if pd.get("team") == task["true_team"] else 0.0)

        # Info request — novel scored mechanic
        if self.info_request_hit:
            ir = w["info_request"] * 1.0     # asked for the right thing
        elif self.info_requested:
            ir = w["info_request"] * 0.30    # asked something, not the key evidence
        else:
            ir = w["info_request"] * -0.10   # never requested info — missed the mechanism

        # Response quality
        r = w["response_quality"] * _response_score(pd.get("response", ""), task)

        # Reasoning quality
        rsn = w["reasoning_quality"] * _reasoning_score(self.reasons_given, "hard")

        total = c + p + t + ir + r + rsn
        if self.sla_breached:
            total *= 0.80
            total = max(0.0, min(1.0, total))
            total = safe_score(total)
            return round(total, 4)

    def report(self, final_obs: Observation, task: Dict[str, Any]) -> dict:
        pd = final_obs.partial_decisions
        return {
            "task":       task["id"],
            "difficulty": "expert",
            "score":      self.score(final_obs, task),
            "details": {
                "classification":           pd.get("classification"),
                "expected_classification":  task["true_category"],
                "classification_correct":   pd.get("classification") == task["true_category"],
                "priority_given":           pd.get("priority"),
                "expected_priority":        task["true_priority"],
                "priority_score":           _priority_score(pd.get("priority"), task["true_priority"]),
                "chose_high_priority_wrong": self.chose_high_priority,
                "team_assigned":            pd.get("team"),
                "expected_team":            task["true_team"],
                "team_correct":             pd.get("team") == task["true_team"],
                "info_requested":           self.info_requested,
                "info_request_on_target":   self.info_request_hit,
                "response_quality":         _response_score(pd.get("response", ""), task),
                "response_blameless":       not any(
                    n in (pd.get("response") or "").lower()
                    for n in ["marcus", "priya", "tom"]
                ),
                "reasoning_steps":          self.reasons_given,
                "sla_breached":             self.sla_breached,
                "steps_taken":              self.steps,
                "grader_notes":             task["task_context"].get("grader_notes", ""),
            }
        }
