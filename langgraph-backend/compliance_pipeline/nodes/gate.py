"""`gate`: decide which Stage 2 checklist items apply to this call.

Pure code, no LLM. Evaluates each Stage 2 item's `applies_when` against
`call_intent`; every condition on an item must hold (AND):

- `always: true` -- always applies.
- `call_intent: [...]` -- applies when the call's category is in the list.
  If the category could not be determined (`None`), the item still applies:
  a failed classification must never hide a check.

An item with no conditions applies. Writes `applicable_items` and a
`NOT_APPLICABLE` `ItemResult` for each gated-out item, citing the
`call_intent` evidence that decided it.

Not wired into `graph.py` yet: the graph currently sends every Stage 2 item
to `check_item`.
"""
from __future__ import annotations

from compliance_pipeline.config import active_checklist
from compliance_pipeline.schemas import ChecklistItem, ItemResult, Status
from compliance_pipeline.state import ComplianceState


def evaluate(item: ChecklistItem, call_intent: dict | None) -> tuple[bool, str]:
    """Whether `item` applies, and a one-line reason for reviewers."""
    category = (call_intent or {}).get("category")
    for key, value in item.applies_when.items():
        if key == "always":
            if not value:
                return False, "applies_when.always is false"
        elif key == "call_intent":
            allowed = list(value or [])
            if category is None:
                continue  # undetermined intent: run the check rather than hide it
            if category not in allowed:
                return False, f"call intent is {category}; applies only to {', '.join(allowed)}"
    if category is None and "call_intent" in item.applies_when:
        return True, "call intent undetermined; applied so the check is not skipped"
    return True, "conditions met"


def gate_node(state: ComplianceState) -> dict:
    call_intent = state.get("call_intent")
    applicable: list[str] = []
    gated_out: list[ItemResult] = []
    for item in active_checklist(state):
        if item.stage != 2:
            continue
        applies, reason = evaluate(item, call_intent)
        if applies:
            applicable.append(item.id)
        else:
            gated_out.append(ItemResult(
                item_id=item.id,
                status=Status.NOT_APPLICABLE,
                confidence=(call_intent or {}).get("confidence", 0.0),
                evidence=(call_intent or {}).get("evidence", []),
                reasoning=f"Gated out: {reason}.",
            ))
    return {"applicable_items": applicable, "item_results": gated_out}
