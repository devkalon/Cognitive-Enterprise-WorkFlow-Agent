"""
CEWA — Cognitive Enterprise Workflow Agent
Main Environment — Full OpenEnv spec: step() / reset() / state()

The agent acts as a cognitive enterprise workflow manager.
Each episode presents a realistic enterprise ticket scenario.
The agent must classify, prioritize, assign, and resolve under SLA.
"""
from __future__ import annotations
import copy
import uuid
from typing import Any, Dict, List, Optional, Tuple

from env.state import (
    Observation, Action, Reward, StepInfo, SLAStatus, PartialDecisions
)
from env.rewards import RewardCalculator
from tasks import get_task, load_tasks


# Valid action values per type
VALID_CATEGORIES = {"billing", "bug", "access", "network", "security", "hr", "general"}
VALID_TEAMS      = {"finance", "tech", "security", "hr", "management"}
VALID_PRIORITIES = {"1", "2", "3", "4", "5"}


class CEWAEnv:
    """
    OpenEnv-compliant Cognitive Enterprise Workflow Agent Environment.

    Each episode: agent receives a realistic enterprise ticket scenario
    and must make a sequence of workflow decisions (classify → prioritize
    → assign → respond) within SLA constraints.

    3 task difficulties: easy → medium → hard.
    Dense reward signal every step.
    """

    METADATA = {
        "name": "cognitive-enterprise-workflow-agent",
        "version": "1.0.0",
        "description": (
            "Enterprise IT workflow triage environment. Agent classifies tickets, "
            "sets priority, routes to correct team, decides on escalation, and "
            "drafts responses — all under SLA constraints."
        ),
        "action_space": "discrete + parameterized (ActionType + value string)",
        "observation_space": "structured Pydantic Observation model",
        "reward_range": (-1.0, 1.0),
        "tasks": ["task1_basic_triage", "task2_incident_triage", "task3_complex_orchestration"],
    }

    def __init__(
        self,
        task_id: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.task_id = task_id
        self.seed = seed
        self._reward_calc = RewardCalculator()
        self._task: Optional[Dict[str, Any]] = None
        self._obs: Optional[Observation] = None
        self._step_count = 0
        self._done = False

    # ── OpenEnv API ───────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Observation:
        """Reset environment. Returns initial observation."""
        if seed is not None:
            self.seed = seed

        # Load task
        if self.task_id:
            self._task = get_task(self.task_id)
        else:
            import random
            rng = random.Random(self.seed)
            self._task = rng.choice(load_tasks())

        self._step_count = 0
        self._done = False

        self._obs = Observation(
            task_id=self._task["id"],
            task_description=self._task["description"],
            difficulty=self._task["difficulty"],
            step=0,
            episode_length=self._task["episode_length"],
            history=[],
            partial_decisions={},
            sla=SLAStatus(steps_remaining=self._task["sla_steps"], breached=False),
            similar_past_tickets=self._task.get("similar_past_tickets", []),
            team_availability=self._task.get("team_availability", {}),
            urgency_signals=self._task.get("urgency_signals", []),
            task_context=self._task.get("task_context", {}),
        )
        return copy.deepcopy(self._obs)

    def step(self, action: Action) -> Tuple[Observation, float, bool, StepInfo]:
        """Execute one action. Returns (observation, reward, done, info)."""
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")
        if self._obs is None:
            raise RuntimeError("Call reset() before step().")

        prev_obs = copy.deepcopy(self._obs)
        events: List[str] = []

        # Validate
        action_valid, action_error = self._validate_action(action)
        if action_valid:
            self._execute_action(action, events)
        else:
            events.append(f"INVALID: {action_error}")

        # Tick SLA
        self._obs.sla.steps_remaining -= 1
        if self._obs.sla.steps_remaining <= 0 and not self._obs.partial_decisions.get("response"):
            self._obs.sla.breached = True
            events.append("SLA_BREACHED: no response drafted before deadline")
        elif self._obs.sla.steps_remaining == 1:
            events.append("SLA_WARNING: 1 step remaining to draft response")

        # Compute reward
        reward_obj = self._reward_calc.compute(
            task=self._task,
            prev_obs=prev_obs,
            action=action,
            next_obs=self._obs,
            events=events,
            action_valid=action_valid,
        )

        self._step_count += 1
        self._obs.step = self._step_count

        # Done conditions
        done = (
            self._step_count >= self._task["episode_length"]
            or self._is_complete()
        )
        self._done = done

        step_reward = reward_obj.total

        if done:
            final = self._reward_calc.compute_episode_final_reward(self._task, self._obs)
            step_reward += final
            reward_obj.breakdown["episode_final"] = final

        task_score = self._compute_task_score() if done else None

        info = StepInfo(
            step=self._step_count,
            done=done,
            truncated=False,
            reward_breakdown=reward_obj.breakdown,
            events=events,
            task_score=task_score,
        )

        step_reward = max(-1.0, min(1.0, step_reward))  # clamp to spec range [-1, 1]
        return copy.deepcopy(self._obs), round(step_reward, 4), done, info

    def state(self) -> Observation:
        """Return current state without stepping."""
        if self._obs is None:
            raise RuntimeError("Call reset() first.")
        return copy.deepcopy(self._obs)

    # ── Validation & Execution ────────────────────────────────────────────

    def _validate_action(self, action: Action) -> Tuple[bool, str]:
        if action.action_type == "classify":
            if action.value not in VALID_CATEGORIES:
                return False, f"invalid category '{action.value}'. Valid: {VALID_CATEGORIES}"
        elif action.action_type == "prioritize":
            if action.value not in VALID_PRIORITIES:
                return False, f"invalid priority '{action.value}'. Must be 1–5"
        elif action.action_type == "assign":
            if action.value not in VALID_TEAMS:
                return False, f"invalid team '{action.value}'. Valid: {VALID_TEAMS}"
        elif action.action_type == "respond":
            if not action.value or len(action.value.strip()) < 10:
                return False, "response must be at least 10 characters"
        return True, ""

    def _execute_action(self, action: Action, events: List[str]):
        obs = self._obs
        key = f"{action.action_type}:{action.value}"
        obs.history.append(key)

        if action.action_type == "classify":
            obs.partial_decisions["classification"] = action.value
            events.append(f"CLASSIFIED: {action.value}")

        elif action.action_type == "prioritize":
            obs.partial_decisions["priority"] = int(action.value)
            events.append(f"PRIORITY_SET: {action.value}")

        elif action.action_type == "assign":
            obs.partial_decisions["team"] = action.value
            events.append(f"ASSIGNED: {action.value}")

        elif action.action_type == "respond":
            obs.partial_decisions["response"] = action.value
            events.append(f"RESPONSE_DRAFTED: {action.value[:60]}...")

        elif action.action_type == "escalate":
            obs.partial_decisions["escalated"] = True
            events.append("ESCALATED to management")

        elif action.action_type == "request_info":
            obs.partial_decisions["info_requested"] = True
            events.append(f"INFO_REQUESTED: {action.value}")

        # Track substantive reasoning for deterministic scoring
        if action.reason and len(action.reason.strip()) >= 15:
            obs.partial_decisions["_reasons_given"] = obs.partial_decisions.get("_reasons_given", 0) + 1

    def _is_complete(self) -> bool:
        """Episode complete when all 4 required decisions made."""
        pd = self._obs.partial_decisions
        return all(pd.get(k) for k in ["classification", "priority", "team", "response"])

    # ── Scoring ───────────────────────────────────────────────────────────

    def _compute_task_score(self) -> float:
        """Final 0.0–1.0 task score."""
        task = self._task
        pd = self._obs.partial_decisions
        weights = task["task_context"].get("score_weights", {})

        score = 0.0

        # Classification
        w = weights.get("classification", 0.25)
        if pd.get("classification") == task["true_category"]:
            score += w

        # Priority (within 1 is acceptable)
        w = weights.get("priority", 0.20)
        try:
            diff = abs(int(pd.get("priority", 0)) - task["true_priority"])
            if diff == 0:
                score += w
            elif diff == 1:
                score += w * 0.5
        except (ValueError, TypeError):
            pass

        # Team assignment
        w = weights.get("team_assignment", 0.25)
        if pd.get("team") == task["true_team"]:
            score += w

        # Response quality
        w = weights.get("response_quality", 0.20)
        response = (pd.get("response") or "").lower()
        key_words = task.get("key_resolution_words", [])
        matches = sum(1 for kw in key_words if kw in response)
        if matches >= 2:
            score += w
        elif matches == 1:
            score += w * 0.5

        # Escalation decision (task2 + task3)
        w = weights.get("escalation_decision", 0)
        if w > 0:
            requires = task["task_context"].get("requires_escalation", False)
            escalated = pd.get("escalated", False)
            if requires and escalated:
                score += w
            elif not requires and not escalated:
                score += w

        # Reasoning quality — deterministic: fraction of steps where agent gave a substantive reason
        w = weights.get("reasoning_quality", 0)
        if w > 0:
            difficulty = self._task.get("difficulty", "easy")
            needed = {"easy": 1, "medium": 2, "hard": 3}.get(difficulty, 1)
            reasons_given = self._obs.partial_decisions.get("_reasons_given", 0)
            reasoning_ratio = min(1.0, reasons_given / max(needed, 1))
            score += w * reasoning_ratio

        # SLA penalty
        if self._obs.sla.breached:
            score *= 0.75

        return round(max(0.0, min(1.0, score)), 4)
