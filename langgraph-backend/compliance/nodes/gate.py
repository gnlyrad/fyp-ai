"""`gate`: decide which Stage 2 checklist items apply to this call.

Final intended behaviour (CLAUDE.md): evaluate each Stage 2 item's
`applies_when` (`always`, `call_intent: [...]`, `has_orders: true`,
`product_discussed: true`) against `call_intent`/`orders`/`product_context`,
write `applicable_items`, and write `NOT_APPLICABLE` `ItemResult`s for the
rest.

Step-0 stub (simple-but-real): every Stage 2 item is applicable, since
`applies_when` conditions cannot yet be evaluated meaningfully without the
Stage 1 nodes implemented (they are still stubs). This is enough to exercise
the `Send` fan-out end to end.
"""
from __future__ import annotations

from compliance.config import load_checklist
from compliance.state import ComplianceState


def gate_node(state: ComplianceState) -> dict:
    applicable = [item.id for item in load_checklist() if item.stage == 2]
    return {"applicable_items": applicable}
