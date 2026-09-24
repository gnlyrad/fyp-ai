"""Prompt for `nodes/decide.py`.

TODO(step 9): two uses -- (1) escalation, LLM only when no escalation_rules
rule fires but some result is PARTIAL/CANNOT_DETERMINE; (2) post-call
actions, listing `action_type` (place_order, cancel_order, send_documents,
obtain_information, raise_case) + description from confirmed orders and the
transcript. Ownership is filled in code from ownership.yaml, never by the LLM.
"""
from __future__ import annotations

SYSTEM_ESCALATION = "TODO: escalation-on-unclear-results system prompt."
SYSTEM_ACTIONS = "TODO: post-call action proposal system prompt."


def build_escalation_user(results: list[dict]) -> str:
    """TODO: format Stage 2 results for the fallback escalation call."""
    raise NotImplementedError


def build_actions_user(orders: list[dict], transcript: str) -> str:
    """TODO: format orders + transcript for the post-call action call."""
    raise NotImplementedError
