"""
CEWA — Cognitive Enterprise Workflow Agent
State Models: Observation, Action, Reward, StepInfo

Typed Pydantic models — required for OpenEnv spec compliance.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# ── Enums ──────────────────────────────────────────────────────────────────

TicketCategory  = Literal["billing", "bug", "access", "network", "security", "hr", "general"]
TicketPriority  = Literal[1, 2, 3, 4, 5]          # 5 = critical
TicketStatus    = Literal["open", "in_progress", "pending_info", "escalated", "resolved"]
TeamName        = Literal["finance", "tech", "security", "hr", "management"]

ActionType = Literal[
    "classify",       # set ticket category
    "prioritize",     # set urgency 1–5
    "assign",         # route to team
    "respond",        # draft resolution response
    "escalate",       # escalate to management
    "request_info",   # ask requester for more detail
]

# ── Sub-models ─────────────────────────────────────────────────────────────

class Ticket(BaseModel):
    ticket_id: str
    description: str
    requester: str = "user@company.com"
    true_category: TicketCategory
    true_priority: int = Field(ge=1, le=5)
    true_team: TeamName
    sla_steps: int = 5             # steps until SLA breach


class PartialDecisions(BaseModel):
    """Tracks what the agent has decided so far in this episode."""
    classification: Optional[TicketCategory] = None
    priority: Optional[int] = None
    team: Optional[TeamName] = None
    response: Optional[str] = None
    escalated: bool = False
    info_requested: bool = False


class SLAStatus(BaseModel):
    steps_remaining: int
    breached: bool = False


# ── Main Observation ───────────────────────────────────────────────────────

class Observation(BaseModel):
    # Task context
    task_id: str
    task_description: str
    difficulty: Literal["easy", "medium", "hard"]

    # Episode progress
    step: int = 0
    episode_length: int = 6
    history: List[str] = Field(default_factory=list)

    # Current state
    partial_decisions: Dict[str, Any] = Field(default_factory=dict)
    sla: SLAStatus = Field(default_factory=lambda: SLAStatus(steps_remaining=5))

    # Context signals
    similar_past_tickets: List[str] = Field(default_factory=list)
    team_availability: Dict[str, bool] = Field(default_factory=dict)
    urgency_signals: List[str] = Field(default_factory=list)

    # Task metadata (injected per task)
    task_context: Dict[str, Any] = Field(default_factory=dict)


# ── Action ─────────────────────────────────────────────────────────────────

class Action(BaseModel):
    action_type: ActionType
    value: str
    reason: Optional[str] = None   # agent's stated reasoning — used in grading


# ── Reward ─────────────────────────────────────────────────────────────────

class Reward(BaseModel):
    total: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)


# ── Step Info ──────────────────────────────────────────────────────────────

class StepInfo(BaseModel):
    step: int = 0
    done: bool = False
    truncated: bool = False
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    events: List[str] = Field(default_factory=list)
    task_score: Optional[float] = None
