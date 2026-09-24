"""`call_intent`: classify the call as enquiry / execution_only / advisory.

Final intended behaviour (CLAUDE.md): LLM call over the formatted
transcript. A call can move from enquiry to order mid-call, so classify the
most regulated intent reached anywhere in the call. Runs in parallel with
`extraction`, both fed by `role_inference`.

Step-0 stub: always returns no intent.
"""
from __future__ import annotations

from compliance.state import ComplianceState


def call_intent_node(state: ComplianceState) -> dict:
    return {"call_intent": None}
