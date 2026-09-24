"""`extraction`: pull orders and entities out of the transcript.

Final intended behaviour (CLAUDE.md): LLM returns a list of orders, marking
corrections to earlier instructions as `superseded` and whether each order
was explicitly confirmed by the client. `normalise` then fills in derived
fields. Runs in parallel with `call_intent`, both fed by `role_inference`.

Step-0 stub: always returns no orders.
"""
from __future__ import annotations

from compliance.state import ComplianceState


def extraction_node(state: ComplianceState) -> dict:
    return {"orders": []}
