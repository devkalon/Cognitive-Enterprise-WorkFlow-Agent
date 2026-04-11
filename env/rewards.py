"""
CEWA Reward Calculator — v2

Improvements over v1:
- Graduated scoring (exact / close / far) for every decision
- Response quality uses BOTH must_include AND must_not_include lists
- Escalation logic is bidirectional (correct escalation OR correct non-escalation)
- Trajectory bonus: reward for efficient decision ordering (classify first)
- SLA warning step gives partial penalty before full breach
- Wrong team on security task gets larger penalty (safety-critical error)
- Episode final: all-correct bonus, completeness penalty, SLA discipline
- All values documented with rationale
"""
from __future__ import annotations
from typing import Dict, List, Any
from env.state import Observation, Action, Reward


class RewardCalculator:

    # ── Step-level reward values ───────────────────────────────────────
    # Classification
    R_CLASS_CORRECT   = +0.30
    R_CLASS_WRONG     = -0.22   # slightly sharper than v1

    # Priority (graduated)
    R_PRI_EXACT       = +0.20
    R_PRI_OFF1        = +0.08   # close — partial credit
    R_PRI_OFF2        = -0.08   # noticeable error
    R_PRI_FAR         = -0.18   # large error

    # Team assignment
    R_TEAM_CORRECT    = +0.25
    R_TEAM_WRONG      = -0.20
    R_TEAM_SECURITY_WRONG = -0.30   # extra penalty — wrong team on security incident = safety-critical

    # Response quality
    R_RESP_STRONG     = +0.20   # ≥2 must_include words, no forbidden words
    R_RESP_PARTIAL    = +0.08   # 1 must_include word
    R_RESP_WEAK       = -0.12   # 0 must_include words
    R_RESP_FORBIDDEN  = -0.10   # used forbidden word (e.g. "invoice" in security response)

    # Escalation
    R_ESC_CORRECT     = +0.15   # escalated when required
    R_ESC_CORRECT_SKIP = +0.10  # correctly did NOT escalate when not needed
    R_ESC_MISSED      = -0.18   # should have escalated, didn't
    R_ESC_UNNECESSARY = -0.12   # escalated when not needed

    # Request info
    R_INFO_USEFUL     = +0.08   # ambiguous (hard) task
    R_INFO_WASTE      = -0.04   # clear (easy) task

    # Reasoning quality
    R_REASON_GOOD     = +0.06   # reason ≥ 20 chars, substantive
    R_REASON_MINIMAL  = +0.02   # reason ≥ 8 chars

    # Ordering efficiency
    R_ORDER_BONUS     = +0.05   # classified before assigning — logical ordering

    # Negative behaviors
    R_REPEATED        = -0.08   # exact same action:value repeated
    R_INVALID         = -0.25

    # SLA
    R_SLA_WARNING     = -0.08   # last step before breach (warning)
    R_SLA_BREACH      = -0.28   # breach confirmed

    # ── Episode-final values ───────────────────────────────────────────
    R_ALL_CORRECT_BONUS = +0.25  # all 4 decisions correct
    R_COMPLETE_BONUS    = +0.10  # all decisions made (regardless of correctness)
    R_MISSING_DECISION  = -0.12  # per missing required decision
    R_SLA_KEPT          = +0.08  # episode ended without SLA breach

    def compute(
        self,
        task: Dict[str, Any],
        prev_obs: Observation,
        action: Action,
        next_obs: Observation,
        events: List[str],
        action_valid: bool,
    ) -> Reward:
        bd: Dict[str, float] = {}

        if not action_valid:
            bd["invalid_action"] = self.R_INVALID
            return Reward(total=self.R_INVALID, breakdown=bd)

        # ── Repeated action ────────────────────────────────────────────
        key = f"{action.action_type}:{action.value}"
        if prev_obs.history.count(key) >= 1:
            bd["repeated_action"] = self.R_REPEATED

        # ── Reasoning quality ──────────────────────────────────────────
        reason_len = len((action.reason or "").strip())
        if reason_len >= 20:
            bd["reasoning_good"] = self.R_REASON_GOOD
        elif reason_len >= 8:
            bd["reasoning_minimal"] = self.R_REASON_MINIMAL

        # ── Ordering efficiency (classify before assign) ───────────────
        pd = prev_obs.partial_decisions
        if action.action_type == "assign" and pd.get("classification"):
            bd["order_bonus"] = self.R_ORDER_BONUS

        # ── Action-specific rewards ────────────────────────────────────
        if action.action_type == "classify":
            if action.value == task["true_category"]:
                bd["correct_classification"] = self.R_CLASS_CORRECT
            else:
                bd["wrong_classification"] = self.R_CLASS_WRONG

        elif action.action_type == "prioritize":
            try:
                diff = abs(int(action.value) - task["true_priority"])
                if diff == 0:
                    bd["priority_exact"] = self.R_PRI_EXACT
                elif diff == 1:
                    bd["priority_off1"] = self.R_PRI_OFF1
                elif diff == 2:
                    bd["priority_off2"] = self.R_PRI_OFF2
                else:
                    bd["priority_far"] = self.R_PRI_FAR
            except (ValueError, TypeError):
                bd["priority_invalid"] = self.R_INVALID

        elif action.action_type == "assign":
            is_security_task = task["true_category"] == "security"
            if action.value == task["true_team"]:
                bd["correct_team"] = self.R_TEAM_CORRECT
            elif is_security_task:
                # Extra penalty: wrong team on security = dangerous
                bd["wrong_team_security"] = self.R_TEAM_SECURITY_WRONG
            else:
                bd["wrong_team"] = self.R_TEAM_WRONG

        elif action.action_type == "respond":
            rv = action.value.lower()
            must_inc  = task.get("response_must_include", [])
            must_not  = task.get("response_must_not_include", [])
            matches   = sum(1 for w in must_inc if w in rv)
            forbidden = sum(1 for w in must_not if w in rv)

            if forbidden > 0:
                bd["response_forbidden_word"] = self.R_RESP_FORBIDDEN * forbidden
            if matches >= 2:
                bd["response_strong"] = self.R_RESP_STRONG
            elif matches == 1:
                bd["response_partial"] = self.R_RESP_PARTIAL
            else:
                bd["response_weak"] = self.R_RESP_WEAK

        elif action.action_type == "escalate":
            requires = task.get("requires_escalation", False)
            if requires:
                bd["correct_escalation"] = self.R_ESC_CORRECT
            else:
                bd["unnecessary_escalation"] = self.R_ESC_UNNECESSARY

        elif action.action_type == "request_info":
            if task["difficulty"] == "hard":
                bd["info_request_useful"] = self.R_INFO_USEFUL
            else:
                bd["info_request_waste"] = self.R_INFO_WASTE

        # ── SLA signals ────────────────────────────────────────────────
        if next_obs.sla.breached:
            bd["sla_breach"] = self.R_SLA_BREACH
        elif next_obs.sla.steps_remaining == 1:
            bd["sla_warning"] = self.R_SLA_WARNING

        total = sum(bd.values())
        total = max(-1.0, min(1.0, total))
        return Reward(total=round(total, 4), breakdown=bd)

    def compute_episode_final_reward(
        self, task: Dict[str, Any], final_obs: Observation
    ) -> float:
        pd   = final_obs.partial_decisions
        bd   = {}

        # Completeness
        required = ["classification", "priority", "team", "response"]
        missing  = [k for k in required if not pd.get(k)]
        if not missing:
            bd["complete_bonus"] = self.R_COMPLETE_BONUS
        else:
            bd["missing_decisions"] = self.R_MISSING_DECISION * len(missing)

        # All-correct bonus
        try:
            pri_diff = abs(int(pd.get("priority", 0)) - task["true_priority"])
        except (ValueError, TypeError):
            pri_diff = 99
        all_correct = (
            pd.get("classification") == task["true_category"]
            and pd.get("team") == task["true_team"]
            and pri_diff <= 1
            and bool(pd.get("response"))
        )
        if all_correct:
            bd["all_correct_bonus"] = self.R_ALL_CORRECT_BONUS

        # Escalation completeness check
        requires_esc = task.get("requires_escalation", False)
        escalated    = pd.get("escalated", False)
        if requires_esc and not escalated:
            bd["missed_escalation"] = self.R_ESC_MISSED
        elif not requires_esc and not escalated:
            bd["no_esc_correct"] = self.R_ESC_CORRECT_SKIP

        # SLA discipline
        if not final_obs.sla.breached:
            bd["sla_kept"] = self.R_SLA_KEPT

        total = sum(bd.values())
        return max(-1.0, min(1.0, round(total, 4)))
