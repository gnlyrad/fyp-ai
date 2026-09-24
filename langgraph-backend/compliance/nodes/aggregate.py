"""`aggregate`: make sure every checklist item has exactly one final result.

Final intended behaviour (CLAUDE.md): merge `results` with `verified_results`
(a verified entry replaces the matching `item_id` in `results`), ensure every
checklist item id appears exactly once (a missing one becomes
CANNOT_DETERMINE "branch did not return"), and compute a `summary`
(total, counts by status, quality flags, escalation).

Deviation (documented in `state.py`): `results` is an `operator.add`
reducer, so this node cannot overwrite it -- writing to it would append a
second copy of every item instead of replacing. The deduplicated,
gap-filled list is written to the plain-overwrite `final_results` key
instead; `output` reads from `final_results`, not `results`.
"""
from __future__ import annotations

from collections import Counter

from compliance.config import load_checklist
from compliance.schemas import ItemResult, Status
from compliance.state import ComplianceState


def aggregate_node(state: ComplianceState) -> dict:
    results = list(state.get("item_results", []))
    verified = list(state.get("verified_results", []))

    by_item: dict[str, ItemResult] = {r.item_id: r for r in results}
    for v in verified:
        by_item[v.item_id] = v  # verified wins

    all_ids = [item.id for item in load_checklist()]
    final_results: list[ItemResult] = []
    for item_id in all_ids:
        if item_id in by_item:
            final_results.append(by_item[item_id])
        else:
            final_results.append(ItemResult(
                item_id=item_id,
                status=Status.CANNOT_DETERMINE,
                confidence=0.0,
                reasoning="branch did not return",
            ))

    counts = Counter(r.status.value for r in final_results)
    summary = {
        "total": len(final_results),
        "counts": dict(counts),
        "quality_flags": state.get("quality_flags", []),
        "escalation": state.get("escalation"),
    }

    return {"final_results": final_results, "summary": summary}
