"""`verify`: a second, skeptical pass over risky Stage 2 `check_item` results.

Runs once after all `check_item` branches join. It can only UPHOLD or
DOWNGRADE a result (to CANNOT_DETERMINE) -- it never upgrades a result, and
a PASS never becomes a FAIL.

Two passes:

1. Code (no LLM): drop any `EvidenceRef` whose `segment_id` is not in
   `segments`, or whose span is out of bounds. If that leaves a non-absence
   PASS sub-criterion with no evidence, it becomes CANNOT_DETERMINE.
2. LLM, one call per item that has anything worth re-checking: every FAIL
   sub-criterion, every PASS sub-criterion when the item's confidence is
   below `LOW_CONFIDENCE`, and every red flag attached to the item (matched
   to the `answer_type: list` item). NOT_APPLICABLE
   items are skipped. Calls run concurrently (`ThreadPoolExecutor`,
   `MAX_WORKERS`), but results are applied in deterministic item order.

A `not_supported` verdict on a sub-criterion downgrades it to
CANNOT_DETERMINE (never PASS, never a plain FAIL->PASS flip) and swaps in
the LLM's counter-evidence when it resolves. A `not_supported` verdict on a
red flag drops the flag; the sub-criterion it forced to FAIL (same id as
the flag's category) is downgraded too, unless another upheld flag of the
same category remains. Every item touched has its status and evidence
recomputed with the same `rollup_status` helper `check_item` uses.

Because `item_results`/`red_flags` are `operator.add` reducers, replacements
are written to the plain-overwrite `verified_results`/`verified_red_flags`
keys instead (see `state.py`); `aggregate` prefers a verified entry over the
matching `item_results`/`red_flags` entry for the same id.

Never raises: an unexpected error is swallowed into `errors`, leaving
results unverified rather than crashing the graph.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context

from compliance_pipeline import config
from compliance_pipeline.llm import get_llm_response
from compliance_pipeline.nodes._common import (
    apply_rule_warnings,
    evidence_text,
    format_transcript,
    resolve_evidence,
    rollup_status,
)
from compliance_pipeline.prompts.verify import SYSTEM, build_user
from compliance_pipeline.reliability import InvalidStructuredOutput
from compliance_pipeline.schemas import ItemResult, RedFlag, ReviewWarning, Status, VerifyOutput
from compliance_pipeline.state import ComplianceState

# Below this confidence, PASS sub-criteria are re-checked too, not just FAILs.
LOW_CONFIDENCE = 0.7
# Per-item LLM calls run concurrently, bounded so a large fan-out doesn't
# hammer the API.
MAX_WORKERS = 4


# ---------------------------------------------------------------------------
# Step 1: code-level evidence sanity check.
# ---------------------------------------------------------------------------

def _dedupe_refs(refs):
    seen = set()
    out = []
    for ref in refs:
        key = (ref.segment_id, ref.start_char, ref.end_char, ref.verified)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def _clean_evidence(item_id, label, refs, segments_by_id, errors):
    """Drop refs pointing at an unknown segment or an out-of-bounds span."""
    kept = []
    for ref in refs:
        seg = segments_by_id.get(ref.segment_id)
        if seg is None:
            errors.append(
                f"verify[{item_id}]: {label} cites unknown segment {ref.segment_id!r}; evidence dropped"
            )
            continue
        text = seg.get("text", "") or ""
        if not (0 <= ref.start_char < ref.end_char <= len(text)):
            errors.append(
                f"verify[{item_id}]: {label} evidence span out of bounds for segment "
                f"{ref.segment_id!r}; evidence dropped"
            )
            continue
        kept.append(ref)
    return kept


def _is_absence(item_id: str, sub_id: str, checklist=None) -> bool:
    item = config.get_item(item_id, checklist)
    if item is None:
        return False
    for sc in item.sub_criteria:
        if sc.id == sub_id:
            return sc.absence
    return False


def _recompute(item: ItemResult) -> None:
    item.status = rollup_status(item.sub_criteria)
    item.evidence = _dedupe_refs([ref for sub in item.sub_criteria for ref in sub.evidence])


# ---------------------------------------------------------------------------
# Step 2: build LLM targets.
# ---------------------------------------------------------------------------

def _sub_target(item_id: str, sub, cfg_item, status_label: str, segments: list[dict]) -> dict:
    sc_cfg = next((sc for sc in cfg_item.sub_criteria if sc.id == sub.id), None) if cfg_item else None
    text = sc_cfg.text if sc_cfg else sub.id
    return {
        "target_id": sub.id,
        "kind": "sub_criterion",
        "claim": f"{text} -> {status_label}",
        "note": sub.note,
        "evidence_lines": [
            f'[{ref.segment_id}] "{evidence_text(segments, ref)}"' for ref in sub.evidence
        ],
    }


def _redflag_target(idx: int, flag: RedFlag, segments: list[dict]) -> dict:
    return {
        "target_id": f"redflag_{idx}",
        "kind": "red_flag",
        "claim": f"{flag.category}/{flag.severity}: {flag.description}",
        "note": "",
        "evidence_lines": [
            f'[{ref.segment_id}] "{evidence_text(segments, ref)}"' for ref in flag.evidence
        ],
    }


def verify_node(state: ComplianceState) -> dict:
    try:
        return _verify_node(state)
    except Exception as exc:  # noqa: BLE001 - never crash the graph
        return {
            "verified_results": [r.model_copy(update={
                "execution_status": "prediction_failure", "failure_category": "verification_unavailable",
                "failure_stage": "verify", "verified": False,
            }) for r in state.get("item_results", [])],
            "verified_red_flags": list(state.get("red_flags", [])),
            "errors": [f"verify: unexpected error: {type(exc).__name__}: {exc}"],
        }


def _verify_node(state: ComplianceState) -> dict:
    segments = state.get("segments", [])
    segments_by_id = {s.get("id"): s for s in segments}
    checklist = config.active_checklist(state)
    cfg_by_id = {item.id: item for item in checklist}
    item_results = list(state.get("item_results", []))
    red_flags = list(state.get("red_flags", []))
    errors: list[str] = []

    # Work on deep copies -- item_results/red_flags are reducer-owned.
    items: list[ItemResult] = [r.model_copy(deep=True) for r in item_results]
    for item in items:
        item.sub_criteria = [sc.model_copy(deep=True) for sc in item.sub_criteria]
    flags: list[RedFlag] = [f.model_copy(deep=True) for f in red_flags]

    # ---- Step 1: code checks. ------------------------------------------
    changed_item_ids: set[str] = set()
    for item in items:
        item_changed = False
        for sub in item.sub_criteria:
            cleaned = _clean_evidence(item.item_id, f"sub-criterion {sub.id}", sub.evidence, segments_by_id, errors)
            if len(cleaned) != len(sub.evidence):
                sub.evidence = cleaned
                item_changed = True
                if sub.status == Status.PASS and not cleaned and not _is_absence(item.item_id, sub.id, checklist):
                    sub.status = Status.CANNOT_DETERMINE
                    sub.note = "evidence did not survive verification"
        if item_changed:
            _recompute(item)
            apply_rule_warnings(item, cfg_by_id.get(item.item_id))
            changed_item_ids.add(item.item_id)

    for flag in flags:
        flag.evidence = _clean_evidence(
            "redflag", f"{flag.category} red flag", flag.evidence, segments_by_id, errors
        )

    # ---- Step 2: pick LLM targets per item. -----------------------------
    list_item_id = next(
        (it.item_id for it in items
         if (cfg := cfg_by_id.get(it.item_id)) is not None and cfg.answer_type == "list"),
        None,
    )

    targets_by_item: dict[str, list[dict]] = {}
    for item in items:
        cfg_item = cfg_by_id.get(item.item_id)
        if cfg_item is None or cfg_item.stage != 2:
            continue
        if item.status == Status.NOT_APPLICABLE:
            continue

        targets: list[dict] = []
        low_conf = item.confidence < LOW_CONFIDENCE
        for sub in item.sub_criteria:
            if sub.status == Status.FAIL:
                targets.append(_sub_target(item.item_id, sub, cfg_item, "FAIL", segments))
            elif low_conf and sub.status == Status.PASS:
                targets.append(_sub_target(item.item_id, sub, cfg_item, "PASS", segments))

        if item.item_id == list_item_id:
            for idx, flag in enumerate(flags):
                targets.append(_redflag_target(idx, flag, segments))

        if targets:
            targets_by_item[item.item_id] = targets

    # ---- Step 3: one LLM call per item with targets, concurrently. -----
    item_ids_to_call = [it.item_id for it in items if it.item_id in targets_by_item]
    transcript = format_transcript(segments)

    def _call_one(item_id: str):
        cfg_item = cfg_by_id.get(item_id)
        question = cfg_item.question if cfg_item else ""
        def validate(output):
            expected = {target["target_id"] for target in targets_by_item[item_id]}
            actual = [verdict.target_id for verdict in output.verdicts]
            if len(actual) != len(expected) or set(actual) != expected:
                raise InvalidStructuredOutput("Every verification target needs exactly one verdict")
        return get_llm_response(
            "verify", SYSTEM,
            build_user(question, targets_by_item[item_id], transcript),
            VerifyOutput,
            validate=validate,
        )

    llm_results: dict[str, object] = {}
    if item_ids_to_call:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            # Each worker gets its own context, including LangChain callbacks,
            # tracing, and the node's diagnostic collector.
            futures = {item_id: pool.submit(copy_context().run, _call_one, item_id)
                       for item_id in item_ids_to_call}
            for item_id in item_ids_to_call:
                llm_results[item_id] = futures[item_id].result()

    # ---- Step 4: apply verdicts. ----------------------------------------
    items_by_id = {it.item_id: it for it in items}
    verified_item_ids: set[str] = set()
    dropped_flag_idx: set[int] = set()
    dropped_reason_by_category: dict[str, str] = {}

    for item_id in item_ids_to_call:
        llm_result = llm_results[item_id]
        if not llm_result.ok:
            errors.append(f"verify[{item_id}]: {llm_result.error}")
            failed = items_by_id[item_id]
            failed.execution_status = "prediction_failure"
            failed.failure_category = ("incomplete_verification" if llm_result.error == "InvalidStructuredOutput"
                                       else "verification_unavailable")
            failed.failure_stage = "verify"
            changed_item_ids.add(item_id)
            continue

        item = items_by_id[item_id]
        output: VerifyOutput = llm_result.value
        verdict_by_target = {v.target_id: v for v in output.verdicts}
        subs_by_id = {sub.id: sub for sub in item.sub_criteria}

        upheld = 0
        downgraded = 0
        for target in targets_by_item[item_id]:
            tid = target["target_id"]
            verdict = verdict_by_target.get(tid)
            if verdict is None:
                errors.append(f"verify: no verdict for {item_id}/{tid}")
                item.execution_status = "prediction_failure"
                item.failure_category = "incomplete_verification"
                item.failure_stage = "verify"
                upheld += 1
                continue
            if verdict.verdict != "not_supported":
                upheld += 1
                continue
            downgraded += 1

            if target["kind"] == "sub_criterion":
                sub = subs_by_id.get(tid)
                if sub is None:
                    continue
                old_status = sub.status.value
                old_note = sub.note
                old_evidence = list(sub.evidence)
                counter_refs, notes = resolve_evidence(segments, verdict.counter_evidence)
                errors.extend(f"verify[{item_id}]: {n}" for n in notes)
                sub.status = Status.CANNOT_DETERMINE
                sub.note = f"{old_status} not supported on verification: {verdict.reason}".strip()
                if counter_refs:
                    sub.evidence = counter_refs
                if old_status == "FAIL":
                    # Over-reporting by design: a human should still see that
                    # the initial review found a failure here, even though it
                    # didn't hold up on re-check.
                    item.warnings.append(ReviewWarning(
                        sub_criterion_id=sub.id,
                        severity="low",
                        message=(
                            "Initial review found a failure that was not "
                            f"supported on re-check: {old_note}"
                        ),
                        evidence=old_evidence,
                        source="rule",
                    ))
            else:  # red_flag
                idx = int(tid.split("_", 1)[1])
                dropped_flag_idx.add(idx)
                dropped_reason_by_category[flags[idx].category] = verdict.reason

        item.reasoning = f"{item.reasoning} [verify: {upheld} upheld, {downgraded} downgraded]".strip()
        item.verified = item.execution_status == "success"
        verified_item_ids.add(item_id)
        _recompute(item)
        apply_rule_warnings(item, cfg_by_id.get(item_id))

    # ---- Red flags: drop not_supported, downgrade the forced sub. -------
    kept_categories = {flag.category for idx, flag in enumerate(flags) if idx not in dropped_flag_idx}
    if list_item_id is not None:
        list_item = items_by_id[list_item_id]
        for category, reason in dropped_reason_by_category.items():
            if category in kept_categories:
                continue  # another upheld flag of the same category remains
            sub = next((s for s in list_item.sub_criteria if s.id == category), None)
            if sub is None or sub.status != Status.FAIL:
                continue
            sub.status = Status.CANNOT_DETERMINE
            sub.note = f"{sub.note} (red flag not supported on verification: {reason})".strip()
        _recompute(list_item)
        apply_rule_warnings(list_item, cfg_by_id.get(list_item_id))

    final_flags = [flag for idx, flag in enumerate(flags) if idx not in dropped_flag_idx]

    # ---- Assemble output. ------------------------------------------------
    include_ids = changed_item_ids | verified_item_ids
    verified_results = [items_by_id[iid] for iid in items_by_id if iid in include_ids]
    # Preserve original item_results order.
    order = {r.item_id: i for i, r in enumerate(item_results)}
    verified_results.sort(key=lambda r: order.get(r.item_id, 0))

    out: dict = {"verified_results": verified_results, "verified_red_flags": final_flags}
    if errors:
        out["errors"] = errors
    return out
