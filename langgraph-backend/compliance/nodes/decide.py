"""`decide`: escalation, post-call actions, and task ownership.

Final intended behaviour (CLAUDE.md):
1. Escalation: apply `escalation_rules.yaml` in order. LLM only when no rule
   fires but some result is PARTIAL/CANNOT_DETERMINE.
2. Post-call actions: LLM proposes actions (place_order, cancel_order,
   send_documents, obtain_information, raise_case) from confirmed orders and
   the transcript.
3. Ownership: map each `action_type` to a role via `ownership.yaml` in code;
   the LLM never picks the owner.
Also writes the C07/C08/C09 `ItemResult`s (C09 is NOT_APPLICABLE when there
are no actions).

Step-0 stub: no escalation, no actions.
"""
from __future__ import annotations

from compliance.state import ComplianceState


def decide_node(state: ComplianceState) -> dict:
    return {"actions": [], "escalation": None}
