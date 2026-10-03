"""`aggregate`: make sure every checklist item has exactly one final result,
and compute a triage summary for the dashboard (CLAUDE.md "aggregate").

Merges `item_results` with `verified_results` via `nodes/_common.py:
merged_results` (verified wins per `item_id`) rather than duplicating that
merge logic here. Every checklist item appears exactly once, in config
order; a missing branch becomes CANNOT_DETERMINE "branch did not return".
A result whose `item_id` is not a checklist item at all (e.g. a bad
`Send` payload) is dropped from `final_results` and reported in `errors`
instead of being silently included.

Deviation (documented in `state.py`): `item_results`/`red_flags` are
`operator.add` reducers, so this node cannot overwrite them -- writing to
them would append a second copy of every item instead of replacing. The
deduplicated, gap-filled list goes to the plain-overwrite `final_results`
key instead; `output` reads from `final_results`, not `item_results`.
`red_flags` is the same kind of reducer, so the final list
(`verified_red_flags` when `verify` set it -- even if empty -- else the raw
`red_flags`) goes to the plain `final_red_flags` key, which `output` and the
graph's `OutputState` expose.

`summary.escalation` mirrors `decide`'s `Escalation` model
(`{required, hold_transaction, source, rule_ids}`, plus `reasons` for
context) rather than the old ad hoc dict-or-None.

`needs_review` is a triage signal only -- every call goes to a human
reviewer regardless -- true when escalation is required, any Stage 2 item
is FAIL/PARTIAL/CANNOT_DETERMINE, any warning or red flag exists, or a
quality flag was raised. `review_reasons` is a short list of why.
"""
from __future__ import annotations

from collections import Counter

from compliance_pipeline.config import active_checklist, get_item
from compliance_pipeline.schemas import ItemResult, Status
from compliance_pipeline.state import ComplianceState

ALL_STATUSES = [s.value for s in Status]


def aggregate_node(state: ComplianceState) -> dict:
    errors: list[str] = []

    results = list(state.get("item_results", []))
    verified = list(state.get("verified_results", []))

    checklist = active_checklist(state)  # the caller's checklist, or the default
    checklist_ids = {item.id for item in checklist}

    by_item: dict[str, ItemResult] = {}
    for r in results:
        if r.item_id not in checklist_ids:
            errors.append(f"aggregate: dropped result for unknown item id {r.item_id!r}")
            continue
        by_item[r.item_id] = r
    for v in verified:
        if v.item_id not in checklist_ids:
            errors.append(f"aggregate: dropped verified result for unknown item id {v.item_id!r}")
            continue
        by_item[v.item_id] = v  # verified wins

    final_results: list[ItemResult] = []
    for item in checklist:
        if item.id in by_item:
            final_results.append(by_item[item.id])
        else:
            final_results.append(ItemResult(
                item_id=item.id,
                status=Status.CANNOT_DETERMINE,
                confidence=0.0,
                reasoning="branch did not return",
                execution_status="prediction_failure",
                failure_category="missing_prediction",
                failure_stage="aggregate",
            ))

    verified_red_flags = state.get("verified_red_flags")
    final_red_flags = list(verified_red_flags) if verified_red_flags is not None else list(state.get("red_flags", []))

    # ---- Summary. -----------------------------------------------------
    counts = Counter(r.status.value for r in final_results)
    counts_full = {status: counts.get(status, 0) for status in ALL_STATUSES}

    stage_counts: dict[int, dict[str, int]] = {1: {}, 2: {}, 3: {}}
    for r in final_results:
        cfg_item = get_item(r.item_id, checklist)
        stage = cfg_item.stage if cfg_item is not None else None
        if stage not in stage_counts:
            continue
        bucket = stage_counts[stage]
        bucket[r.status.value] = bucket.get(r.status.value, 0) + 1
    for stage in stage_counts:
        stage_counts[stage] = {status: stage_counts[stage].get(status, 0) for status in ALL_STATUSES}

    quality_flags = state.get("quality_flags", []) or []
    call_intent = state.get("call_intent") or {}

    escalation_state = state.get("escalation") or {}
    escalation_summary = {
        "required": bool(escalation_state.get("required", False)),
        "hold_transaction": bool(escalation_state.get("hold_transaction", False)),
        "source": escalation_state.get("source", "none"),
        "rule_ids": list(escalation_state.get("rule_ids", [])),
        "reasons": list(escalation_state.get("reasons", [])),
    }

    actions = state.get("actions", []) or []
    actions_by_owner: dict[str, int] = {}
    for a in actions:
        owner = getattr(a, "owner_role", None) if not isinstance(a, dict) else a.get("owner_role")
        owner = owner or "unassigned"
        actions_by_owner[owner] = actions_by_owner.get(owner, 0) + 1

    red_flags_by_severity: dict[str, int] = {}
    for f in final_red_flags:
        severity = getattr(f, "severity", None) if not isinstance(f, dict) else f.get("severity")
        severity = severity or "unknown"
        red_flags_by_severity[severity] = red_flags_by_severity.get(severity, 0) + 1

    total_warnings = sum(len(r.warnings) for r in final_results)
    items_with_warnings = [r.item_id for r in final_results if r.warnings]
    verified_items = [r.item_id for r in final_results if r.verified]

    summary = {
        "total": len(final_results),
        "counts": counts_full,
        "stage_counts": stage_counts,
        "quality_flags": quality_flags,
        "call_intent": call_intent.get("category"),
        "escalation": escalation_summary,
        "actions": len(actions),
        "actions_by_owner": actions_by_owner,
        "red_flags": len(final_red_flags),
        "red_flags_by_severity": red_flags_by_severity,
        "warnings": total_warnings,
        "items_with_warnings": items_with_warnings,
        "verified_items": verified_items,
        "errors": len(errors) + len(state.get("errors", []) or []),
    }

    # ---- needs_review. --------------------------------------------------
    review_reasons: list[str] = []
    if escalation_summary["required"]:
        review_reasons.append("Escalation required.")

    stage2_bad = [
        r.item_id for r in final_results
        if (get_item(r.item_id, checklist) is not None and get_item(r.item_id, checklist).stage == 2)
        and r.status in (Status.FAIL, Status.PARTIAL, Status.CANNOT_DETERMINE)
    ]
    if stage2_bad:
        review_reasons.append(
            f"Stage 2 item(s) not a clean PASS: {', '.join(stage2_bad)}."
        )

    if total_warnings:
        review_reasons.append(f"{total_warnings} reviewer warning(s) attached.")

    if final_red_flags:
        review_reasons.append(f"{len(final_red_flags)} red flag(s) identified.")

    if quality_flags:
        review_reasons.append(f"Call quality flag(s): {', '.join(quality_flags)}.")

    needs_review = bool(review_reasons)
    summary["needs_review"] = needs_review
    summary["review_reasons"] = review_reasons

    out: dict = {
        "final_results": final_results,
        "final_red_flags": final_red_flags,
        "summary": summary,
    }
    if errors:
        out["errors"] = errors
    return out
