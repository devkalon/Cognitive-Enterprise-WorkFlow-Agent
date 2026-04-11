"""
CEWA Task Definitions — v2
3 tasks: easy → medium → hard

Upgraded features:
- Rich, realistic multi-signal ticket descriptions with timestamps and names
- Explicit ground truth with grader_notes explaining WHY
- Hard distractors on T2/T3 to test genuine reasoning
- response_must_include / response_must_not_include for quality grading
- competing_priorities map on T3 so grader can explain wrong choices
"""
from __future__ import annotations
from typing import List, Dict, Any


def load_tasks() -> List[Dict[str, Any]]:
    return [TASK_1, TASK_2, TASK_3, TASK_4]


def get_task(task_id: str) -> Dict[str, Any]:
    for t in load_tasks():
        if t["id"] == task_id:
            return t
    raise ValueError(f"Unknown task_id: {task_id}")


# ══════════════════════════════════════════════════════════════════════════
# TASK 1 — EASY
# Single domain. Clear signals. No distractors.
# Correct: classify=billing, priority=4, team=finance, respond
# ══════════════════════════════════════════════════════════════════════════

TASK_1 = {
    "id":         "task1_basic_triage",
    "name":       "Billing Dispute — Duplicate Charge",
    "difficulty": "easy",
    "episode_length": 5,

    "description": (
        "TICKET #TKT-8821 | Submitted: 09:14 AM\n"
        "Requester: sarah.k@acme.com | Account Tier: Gold Enterprise\n\n"
        "Subject: URGENT — Charged twice for November subscription\n\n"
        "'I was billed $299 twice on Nov 1st for the same Pro subscription. "
        "My bank statement shows two identical transactions from your company. "
        "I need this refunded immediately — this is the second time this has happened. "
        "If not resolved today I will dispute with my bank and cancel my account.'\n\n"
        "Attachment: bank_statement_nov.pdf (2 matching transactions highlighted)\n"
        "Customer history: 2 prior billing tickets, both resolved by finance team in <4h."
    ),

    "true_category":       "billing",
    "true_priority":       4,
    "true_team":           "finance",
    "requires_escalation": False,

    "key_resolution_words": [
        "billing", "refund", "charge", "payment", "finance",
        "duplicate", "subscription", "resolve", "investigate"
    ],
    "response_must_include":     ["refund", "billing"],
    "response_must_not_include": ["tech", "security", "escalat"],

    "similar_past_tickets": [
        "TKT-8103 | billing | p=4 | finance → RESOLVED 3h: Duplicate charge refunded, billing cycle bug fixed",
        "TKT-7542 | billing | p=3 | finance → RESOLVED 6h: Overcharge on annual renewal",
        "TKT-6991 | billing | p=5 | finance+mgmt → ESCALATED: Enterprise triple-charge, legal threat",
    ],
    "team_availability": {
        "finance": True, "tech": True,
        "security": True, "hr": False, "management": True,
    },
    "urgency_signals": [
        "duplicate charge confirmed on bank statement",
        "Gold Enterprise customer — churn risk",
        "threatens bank dispute + cancellation",
        "second occurrence — pattern emerging",
        "same-day resolution demanded",
    ],
    "sla_steps": 5,

    "task_context": {
        "target_score": 0.85,
        "requires_escalation": False,
        "distractor_signals": [],
        "score_weights": {
            "classification":   0.30,
            "priority":         0.20,
            "team_assignment":  0.25,
            "response_quality": 0.25,
        },
        "grader_notes": (
            "Priority=4 correct (financial impact + Gold churn risk). "
            "Priority=3 gets partial credit. Priority=5 wrong (no SLA breach yet). "
            "Only finance team is correct. No escalation needed yet."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════
# TASK 2 — MEDIUM
# Correlated multi-signal security incident. Tight SLA. Escalation required.
# Distractor: app crash signal could mislead toward "bug" / "tech"
# Correct: classify=security, priority=5, team=security, escalate, respond
# ══════════════════════════════════════════════════════════════════════════

TASK_2 = {
    "id":         "task2_incident_triage",
    "name":       "Security Incident — Credential Attack + Portal Outage",
    "difficulty": "medium",
    "episode_length": 6,

    "description": (
        "INCIDENT ALERT — Auto-generated: 14:37 PM\n"
        "3 correlated signals detected:\n\n"
        "[SIGNAL 1 — IT Monitoring @ 14:31]\n"
        "  847 failed SSH login attempts on prod-api-01 in 8 minutes.\n"
        "  Source IPs: 45.142.x.x, 185.220.x.x (known Tor exit nodes).\n"
        "  Pattern: matches credential stuffing signature in threat DB.\n\n"
        "[SIGNAL 2 — User Reports (14 tickets in 12 minutes)]\n"
        "  London office: compliance portal returning 'Session expired' on every load.\n"
        "  Affected: 14 confirmed users, estimated 60+ in same auth group.\n\n"
        "[SIGNAL 3 — App Monitoring @ 14:31]\n"
        "  compliance-portal: health check FAILING.\n"
        "  Unclear if root cause is auth middleware crash OR forced session wipe.\n\n"
        "Portal down 6 minutes. Finance month-end close in progress.\n"
        "Compliance audit submissions due by 17:00 today (2h 23min)."
    ),

    "true_category":       "security",
    "true_priority":       5,
    "true_team":           "security",
    "requires_escalation": True,

    "key_resolution_words": [
        "security", "breach", "authentication", "incident", "escalat",
        "credentials", "tor", "attack", "soc", "isolat", "investigate"
    ],
    "response_must_include":     ["security", "incident"],
    "response_must_not_include": ["refund", "billing", "hr"],

    "similar_past_tickets": [
        "INC-0089 | security | p=5 | security+mgmt → ESCALATED: Credential stuffing, IP block resolved in 45min",
        "INC-0071 | security | p=5 | security → RESOLVED 2h: Auth spike → session wipe, cred rotation",
        "TKT-9201 | bug | p=3 | tech → RESOLVED: Portal crash = memory leak, NOT security (different pattern)",
        "INC-0055 | security | p=4 | security → RESOLVED: Tor probing, no breach confirmed",
    ],
    "team_availability": {
        "finance": True, "tech": True,   # distractor — tech handles bugs not breaches
        "security": True, "hr": True, "management": True,
    },
    "urgency_signals": [
        "847 failed auth in 8 min — active attack",
        "Tor exit node IPs — external threat actor",
        "credential stuffing pattern — confirmed",
        "60+ users locked out",
        "compliance deadline 17:00 TODAY",
        "finance month-end close in progress",
    ],
    "sla_steps": 4,

    "task_context": {
        "target_score": 0.78,
        "requires_escalation": True,
        "distractor_signals": [
            "app crash signal → could mislead agent to classify as 'bug' and assign 'tech'",
            "tech team available → wrong choice; security owns credential attacks",
        ],
        "score_weights": {
            "classification":      0.25,
            "priority":            0.20,
            "team_assignment":     0.25,
            "escalation_decision": 0.15,
            "response_quality":    0.15,
        },
        "grader_notes": (
            "Security=correct: crash is caused by the auth attack, not a software bug. "
            "Tech team is WRONG — they handle bugs; security handles breach incidents. "
            "Escalation required: potential breach + compliance deadline at risk. "
            "Priority=5 mandatory: active external attack, confirmed pattern, 60+ users."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════
# TASK 3 — HARD
# 3 simultaneous issues, all look urgent. Agent must reason through IMPACT.
# Financial ($142k invoice) and social (CTO starting) are intentional traps.
# Correct: classify=bug (CI/CD), priority=5, team=tech, escalate, respond
# ══════════════════════════════════════════════════════════════════════════

TASK_3 = {
    "id":         "task3_complex_orchestration",
    "name":       "Competing Priority Triage — Queue Overload",
    "difficulty": "hard",
    "episode_length": 7,

    "description": (
        "WORKFLOW QUEUE — 09:02 AM Monday\n"
        "3 issues escalated simultaneously. Identify HIGHEST business impact.\n\n"
        "━━ ISSUE A — Finance ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "TKT-11204 | procurement@acme.com\n"
        "'Invoice #INV-2847 for $142,000 (cloud infra) is 3 days overdue for approval. "
        "AWS reseller emailed twice threatening service suspension by EOD today. "
        "Our AWS bill is still running — we could lose cloud access by tonight.'\n\n"
        "━━ ISSUE B — HR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "TKT-11205 | hr@acme.com\n"
        "'New CTO (James Reeves, $380k/yr) starts TODAY at 10:00 AM — 58 min away. "
        "IT provisioning INCOMPLETE: no laptop, no email, no GitHub, no Slack. "
        "CEO is asking for a status update every 15 minutes. Very embarrassing.'\n\n"
        "━━ ISSUE C — Engineering ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "TKT-11206 | devops@acme.com\n"
        "'PRODUCTION CI/CD PIPELINE BROKEN — deploy key auto-rotated at midnight, "
        "new key not propagated to pipeline. Impact: 6 teams, 47 developers CANNOT "
        "deploy to production or staging. 3 critical hotfixes queued (2 are customer-impacting). "
        "Finance estimates $8,200/hr revenue loss. Outage started 00:05 AM — NOW 9 HOURS. "
        "Total estimated loss so far: $73,800.'\n\n"
        "ACTION REQUIRED: Select the HIGHEST priority issue. Classify it. Assign it. "
        "Escalate if needed. Draft a response that justifies your priority decision."
    ),

    "true_category":       "bug",
    "true_priority":       5,
    "true_team":           "tech",
    "requires_escalation": True,

    "key_resolution_words": [
        "pipeline", "ci/cd", "cicd", "deploy", "devops",
        "engineering", "production", "hotfix", "outage",
        "revenue", "tech", "key", "propagat"
    ],
    "response_must_include":     ["pipeline", "deploy"],
    "response_must_not_include": ["invoice", "onboarding", "cto"],

    "similar_past_tickets": [
        "INC-0101 | bug | p=5 | tech+mgmt → ESCALATED: CI/CD key expiry blocked 5 teams 6h, $52k loss",
        "INC-0098 | bug | p=5 | tech → RESOLVED 45min: Deploy key rotation not propagated — same pattern",
        "TKT-9988 | billing | p=4 | finance → RESOLVED: $95k invoice, vendor gave 48h extension after call",
        "TKT-9121 | hr | p=3 | hr+tech → RESOLVED 2h: Executive onboarding with temp credentials",
    ],
    "team_availability": {
        "finance":    True,    # distractor — Issue A team
        "tech":       True,    # CORRECT — Issue C team
        "security":   False,
        "hr":         True,    # distractor — Issue B team
        "management": True,
    },
    "urgency_signals": [
        "ISSUE C: $73,800 revenue lost — 9h outage @ $8,200/hr",
        "ISSUE C: 47 developers blocked, 3 customer-impacting hotfixes queued",
        "ISSUE C: customer-facing bugs live in production right now",
        "ISSUE A: $142k invoice overdue — vendor threatens EOD suspension",
        "ISSUE B: CTO starts in 58 min — CEO watching",
        "CRITICAL: reason through financial + operational impact, not surface urgency",
    ],
    "sla_steps": 4,

    "task_context": {
        "target_score": 0.72,
        "requires_escalation": True,
        "competing_priorities": {
            "bug (CI/CD)":  "CORRECT — $73.8k lost, 47 devs blocked, 9h outage, customers affected",
            "billing":      "WRONG — Vendor hasn't suspended yet; 48h extension typical; fixable in 1h call",
            "hr":           "WRONG — Embarrassing but zero operational/financial impact",
            "security":     "WRONG — No security signals in this ticket",
            "access":       "PARTIALLY correct reasoning but wrong classification",
        },
        "distractor_signals": [
            "$142,000 invoice → large number anchors agent on billing",
            "CTO + CEO pressure → social urgency, not business impact",
            "'cloud access by tonight' → vague threat, not imminent",
        ],
        "score_weights": {
            "classification":      0.20,
            "priority":            0.20,
            "team_assignment":     0.25,
            "escalation_decision": 0.15,
            "response_quality":    0.10,
            "reasoning_quality":   0.10,
        },
        "grader_notes": (
            "Agent must reason PAST the $142k number and CEO pressure. "
            "CI/CD outage has: confirmed hourly revenue loss ($8.2k/hr), most people blocked (47), "
            "longest duration (9h), customer-facing bugs live. "
            "Invoice: vendor typically gives 24-48h extension when called; no actual suspension yet. "
            "CTO onboarding: embarrassing, fixable with temp credentials in <30min, zero $ impact. "
            "Classify=bug, team=tech, priority=5, escalate=True (9h outage + CTO starting + $74k)."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════
# TASK 4 — EXPERT (beyond hard)
# Post-mortem contradiction analysis. Three team leads submitted retrospective
# accounts of last night's outage — but their timelines contradict each other
# and one team is clearly covering up a deployment error.
# The agent must identify the real root cause from conflicting evidence,
# classify accurately despite social pressure, request the one piece of
# missing evidence that resolves the contradiction, then draft a blameless
# post-mortem that names the actual failure mode without blaming individuals.
#
# Novel mechanics:
#  - request_info is MANDATORY and SCORED (not just optional)
#  - "blameless" framing: response_must_not_include has person names
#  - Three conflicting narratives with internal consistency errors
#  - Distractor: the loudest complainant (infra team) sounds most credible
#    but their timeline is mathematically impossible
#  - Correct: classify=bug, priority=3 (outage resolved, now forensic),
#    team=tech, no escalation, request_info for deploy log hash,
#    respond with blameless RCA
# ══════════════════════════════════════════════════════════════════════════

TASK_4 = {
    "id":         "task4_postmortem_contradiction",
    "name":       "Post-mortem Contradiction — Who Broke Production?",
    "difficulty": "expert",
    "episode_length": 8,

    "description": (
        "POST-MORTEM REQUEST — Submitted: 08:45 AM Tuesday\n"
        "Incident: PROD-DOWN-2024-1119 | Duration: 23 minutes | Now resolved.\n"
        "Three team leads have submitted retrospective accounts. Conflicts detected.\n\n"
        "━━ ACCOUNT A — Infra Lead (Marcus, Infra) ━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "'At 23:47 I received a PagerDuty alert. I checked the load balancer logs\n"
        " immediately — traffic was routing correctly. The problem was clearly in\n"
        " the application layer. I pinged the backend team at 23:48. By 23:52 I\n"
        " had confirmed infra was clean. We restored service at 00:10.'\n\n"
        "━━ ACCOUNT B — Backend Lead (Priya, Backend) ━━━━━━━━━━━━━━━━━━━━\n"
        "'We got Marcus's ping at 23:51. We ran diagnostics — no code errors in\n"
        " production. Our last deploy was 6 hours earlier at 17:45, fully green.\n"
        " We escalated to infra at 23:58 because the DB connection pool was\n"
        " exhausted — that's an infra config issue, not our code.'\n\n"
        "━━ ACCOUNT C — DB Lead (Tom, Database) ━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "'The DB connection pool hit max_connections=200 at exactly 23:47:03.\n"
        " Root cause: a query introduced in deploy d4f9a2 at 23:31 has an N+1\n"
        " loop — under load it opens 40 connections per request instead of 1.\n"
        " Deploy d4f9a2 was pushed by the backend team. I flagged this pattern\n"
        " in code review 3 days ago but the comment was dismissed.'\n\n"
        "━━ AUTOMATED EVIDENCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "PagerDuty: Alert fired 23:47:01\n"
        "DB logs: max_connections breach at 23:47:03\n"
        "Deploy log: d4f9a2 merged to main — timestamp REDACTED (log rotation issue)\n"
        "Backend deploy history (self-reported by Priya): last deploy 17:45\n\n"
        "CONTRADICTION: Tom claims d4f9a2 deployed at 23:31. Priya claims last\n"
        "deploy was 17:45. Marcus's timeline has him 'confirming infra clean' at\n"
        "23:52 — but he says he was pinged at 23:48, and Priya says she received\n"
        "the ping at 23:51. One account is mathematically impossible.\n\n"
        "YOUR TASK: Determine the real root cause. Identify the timeline\n"
        "contradiction. Request the one piece of evidence that definitively\n"
        "resolves the deploy timestamp dispute. Draft a blameless post-mortem\n"
        "that names the failure mode (not the person)."
    ),

    "true_category":       "bug",
    "true_priority":       3,   # outage is resolved — this is forensic/process work
    "true_team":           "tech",
    "requires_escalation": False,

    # The key evidence: the git commit timestamp for d4f9a2
    # If it shows 23:31 → Tom is right, Priya's account is false, backend deployed late
    # Marcus's 4-minute response window (pinged 23:48, confirmed clean 23:52) is plausible
    # But Marcus says he was pinged at 23:48 and Priya says she sent the ping at 23:51
    # → Marcus received a ping 3 minutes before Priya claims to have sent it
    # → Marcus's account is the mathematically impossible one (he invented the timeline)
    # → Marcus is covering: infra config DID have a role (max_connections=200 is low for load)
    # True RCA: N+1 query bug in d4f9a2 (code) + insufficient connection pool ceiling (infra)
    # Blameless framing: "insufficient connection pool config" + "N+1 query pattern"

    "info_request_key": "git commit timestamp for d4f9a2",
    "info_request_keywords": ["d4f9a", "commit", "git", "deploy", "timestamp", "log", "hash"],

    "key_resolution_words": [
        "n+1", "connection pool", "query", "deploy", "d4f9a",
        "blameless", "root cause", "configuration", "pattern", "review"
    ],
    "response_must_include":     ["connection", "query"],
    "response_must_not_include": ["marcus", "priya", "tom", "blame", "fault"],

    "similar_past_tickets": [
        "INC-0077 | bug | p=3 | tech → POST-MORTEM: N+1 query caused pool exhaustion, added query review gate",
        "INC-0061 | bug | p=4 | tech → RESOLVED: max_connections too low for Black Friday load, infra scaled",
        "INC-0044 | bug | p=3 | tech → POST-MORTEM: Deploy at 23:15 not in runbook, caused overnight outage",
    ],
    "team_availability": {
        "finance": False, "tech": True,
        "security": False, "hr": False, "management": True,
    },
    "urgency_signals": [
        "outage resolved — forensic/post-mortem mode, not live incident",
        "three contradictory accounts — one timeline is mathematically impossible",
        "deploy timestamp REDACTED — key evidence missing",
        "Tom flagged N+1 pattern in code review 3 days ago — process failure signal",
        "priority=3: important to resolve for process, not an active emergency",
    ],
    "sla_steps": 6,

    "task_context": {
        "target_score": 0.68,
        "requires_escalation": False,
        "info_request_required": True,   # novel mechanic: request_info is scored
        "contradiction": (
            "Marcus says he was pinged at 23:48. "
            "Priya says she sent the ping at 23:51. "
            "A ping received 3 minutes before it was sent is impossible. "
            "Marcus fabricated his response timeline to appear faster than he was. "
            "This suggests infra was slower to respond and may be covering latency."
        ),
        "true_rca": (
            "Dual failure: (1) N+1 query bug in deploy d4f9a2 opened 40 DB connections/request. "
            "(2) max_connections=200 ceiling too low — infra config not reviewed for traffic growth. "
            "Either fix alone would have prevented the outage."
        ),
        "distractor_signals": [
            "Infra lead (Marcus) sounds most authoritative and detailed — but his timeline is impossible",
            "Priya's 'last deploy was 17:45' sounds calm and confident — but contradicts Tom's deploy log",
            "DB pool exhaustion sounds like an infra problem — but was triggered by a code bug",
        ],
        "score_weights": {
            "classification":      0.15,
            "priority":            0.20,
            "team_assignment":     0.15,
            "info_request":        0.20,   # novel: scored separately
            "response_quality":    0.20,
            "reasoning_quality":   0.10,
        },
        "grader_notes": (
            "Priority=3 correct: outage is over, this is process/forensic work. Priority=4/5 wrong. "
            "info_request scored on whether agent asked for deploy timestamp / git hash for d4f9a2. "
            "Response must name the failure modes (N+1 query, connection pool) not individuals. "
            "Marcus's timeline impossibility is the key insight: received ping before it was sent. "
            "No escalation needed — this is a standard post-mortem, not an active incident."
        ),
    },
}
