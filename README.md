---
title: CEWA OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
=======
# 🧠 CEWA — Cognitive Enterprise Workflow Agent

> **OpenEnv Environment** — AI agent for enterprise IT ticket triage and workflow management

[![OpenEnv](https://img.shields.io/badge/OpenEnv-1.0.0-blue)](https://openenv.dev)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://python.org)

---

## Overview

CEWA is a real-world enterprise workflow management environment where an AI agent acts as a **Cognitive Workflow Manager**. The agent processes realistic enterprise IT ticket scenarios and must:

- **Classify** the ticket category (billing, bug, security, access, hr, etc.)
- **Prioritize** urgency 1–5 based on business impact
- **Assign** to the correct team (finance, tech, security, hr, management)
- **Decide** whether to escalate based on severity
- **Respond** with an appropriate resolution message

All decisions must happen under SLA time constraints with partial information — mirroring what a real IT Operations Manager does daily.

---

## Tasks

### Task 1 — Basic Triage (Easy, 4 steps)
Single billing complaint. Clear signals. Agent classifies correctly, sets priority 4, assigns to finance, drafts response.

**Score:** `classification×0.30 + priority×0.20 + team×0.25 + response×0.25`  
**Baseline score:** ~0.72

### Task 2 — Critical Incident Triage (Medium, 6 steps)
Multi-signal security incident: auth failures + app crash + portal access blocked. Agent must identify security root cause, set priority 5, escalate, and respond under 3-step SLA.

**Score:** `classification×0.25 + priority×0.20 + team×0.25 + escalation×0.15 + response×0.15`  
**Baseline score:** ~0.48

### Task 3 — Complex Orchestration (Hard, 6 steps)
Three simultaneous issues: overdue vendor invoice (finance), executive onboarding (HR), and broken CI/CD pipeline blocking 4 dev teams (engineering). Agent must reason about competing priorities and identify CI/CD as highest impact. Distractors present.

**Score:** `classification×0.20 + priority×0.20 + team×0.25 + escalation×0.15 + response×0.10 + reasoning×0.10`  
**Baseline score:** ~0.31

### Task 4 — Post-mortem Contradiction (Expert, 8 steps)
Three team leads submitted conflicting post-mortem accounts of last night's outage. Their timelines contradict each other — one account is **mathematically impossible** (a ping received before it was sent). The agent must identify the contradiction, classify the real root cause (N+1 query bug compounded by an undersized connection pool), **request the one missing piece of evidence** (git commit timestamp for deploy ), and draft a **blameless** post-mortem that names failure modes without naming individuals.

Novel mechanic:  is **scored at 20% weight** — making it the only task where information gathering is a graded action, not just optional.

**Score:**   
**Baseline score:** ~0.18

---

## Action Space

| Action | Value | Description |
|--------|-------|-------------|
| `classify` | `billing\|bug\|access\|network\|security\|hr\|general` | Set ticket category |
| `prioritize` | `"1"` to `"5"` (5=critical) | Set urgency level |
| `assign` | `finance\|tech\|security\|hr\|management` | Route to team |
| `respond` | any string ≥10 chars | Draft resolution response |
| `escalate` | reason string | Escalate to management |
| `request_info` | question string | Ask requester for more detail |

---

## Reward Signals (dense, every step)

| Signal | Value |
|--------|-------|
| Correct classification | +0.30 |
| Correct priority | +0.20 |
| Correct team | +0.25 |
| Good response (mentions key words) | +0.20 |
| Correct escalation decision | +0.15 |
| Reasoning quality | +0.05 |
| Wrong classification | -0.20 |
| Wrong team | -0.20 |
| SLA breach | -0.25 |
| Invalid action | -0.25 |
| Repeated action | -0.05 |

---

## Observation Space

```python
class Observation(BaseModel):
    task_id: str
    task_description: str        # Full ticket scenario
    difficulty: str              # easy | medium | hard
    step: int
    episode_length: int
    history: List[str]           # Past action:value pairs
    partial_decisions: Dict      # Decisions made so far
    sla: SLAStatus               # steps_remaining, breached
    similar_past_tickets: List[str]   # Context from past cases
    team_availability: Dict[str, bool]
    urgency_signals: List[str]
```

---

## Setup

```bash
# Install
pip install -r requirements.txt

# Run server (HF Space)
python app.py  # http://localhost:7860

# Run baseline inference
API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o-mini HF_TOKEN=sk-... python inference.py

# Docker
docker build -t cewa .
docker run -p 7860:7860 -e API_BASE_URL=... -e MODEL_NAME=... -e HF_TOKEN=... cewa
```

## API

```bash
GET  /health   → {"status": "ok"}
POST /reset    → {"task_id": "task1_basic_triage", "seed": 42}
POST /step     → {"session_id": "...", "action": {"action_type": "classify", "value": "billing", "reason": "..."}}
GET  /state    → ?session_id=...
GET  /tasks    → list all 3 tasks
POST /grade    → ?task_id=task1_basic_triage&seed=42
```

## Example

```python
from env.environment import CEWAEnv
from env.state import Action

env = CEWAEnv(task_id="task2_incident_triage", seed=42)
obs = env.reset()

# Step 1: classify
obs, reward, done, info = env.step(Action(
    action_type="classify", value="security",
    reason="auth failures + access blocked = security incident"
))
print(f"reward={reward}, events={info.events}")

# Step 2: prioritize
obs, reward, done, info = env.step(Action(
    action_type="prioritize", value="5",
    reason="multiple users affected, potential breach"
))
```

## Project Structure

```
cewa-final/
├── app.py              # FastAPI server (HF Space entrypoint)
├── inference.py        # LLM baseline inference (required)
├── openenv.yaml        # OpenEnv metadata spec
├── Dockerfile
├── requirements.txt
├── README.md
├── env/
│   ├── environment.py  # CEWAEnv (step/reset/state)
│   ├── state.py        # Pydantic models
│   └── rewards.py      # Dense reward calculator
├── tasks/__init__.py   # 3 task definitions
└── graders/__init__.py # 3 separate graders (easy/medium/hard)
```

---

## License
MIT
>>>>>>> 8a8e0e2 (Initial commit: CEWA OpenEnv environment)
