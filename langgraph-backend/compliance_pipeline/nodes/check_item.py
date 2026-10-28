"""`check_item`: evaluate one checklist item (one `Send` branch per applicable item).

Loads the item's config, its cached rules (`retrieval.get_rules`), the
transcript, and the call intent; calls the LLM via `llm.get_llm_response`;
then derives the item's overall status from its sub-criteria *in code*
(CLAUDE.md: "Prefer plain code over an LLM whenever the task is
deterministic") rather than trusting the LLM's rollup. `answer_type: list`
items (C06) additionally return `RedFlag`s, and a surviving red flag forces
its matching sub-criterion to FAIL.

A PASS needs a resolved evidence ref, except a configured absence check or
an explicit, complete absence review for an untyped prohibition question.
The latter records the answer's basis without assigning missing metadata.

The LLM cites short verbatim `EvidenceQuote`s; code resolves them to
persisted `EvidenceRef`s via `nodes/_common.py:resolve_evidence`. Rules are
the regulatory standard, never evidence of what happened on the call --
every judgement comes from the transcript alone.

Never raises: an unknown item id, missing segments, or a failed LLM call
all fall back to a `CANNOT_DETERMINE` result plus an `errors` entry instead
of crashing the graph.
"""
from __future__ import annotations

import re

from compliance_pipeline import config
from compliance_pipeline import retrieval
from compliance_pipeline.nodes._common import (
    apply_rule_warnings,
    clean_note,
    format_transcript,
    resolve_evidence,
    rollup_status,
)
from compliance_pipeline.prompts.check_item import SYSTEM, build_user
from compliance_pipeline.llm import get_llm_response
from compliance_pipeline.reliability import InvalidStructuredOutput
from compliance_pipeline.schemas import (
    ChecklistItem,
    CheckItemOutput,
    EvidenceRef,
    ItemResult,
    RedFlag,
    ReviewWarning,
    Status,
    SubCriterionResult,
)
from compliance_pipeline.state import CheckItemTask

# Rule extracts included in the prompt: capped and truncated so the prompt
# stays a bounded size regardless of how many chunks retrieval cached.
MAX_RULES = 8
MAX_RULES_LIST = 12  # answer_type: list (C06) covers six red-flag categories
MAX_RULE_CHARS = 1500

# quality_flags that mean the transcript itself is suspect, so confidence is
# capped even if the LLM was confident -- `verify` re-checks anything at or
# below this.
DEGRADED_CONFIDENCE_CAP = 0.6
DEGRADING_FLAGS = {"low_confidence", "single_speaker"}

_WHITESPACE_RE = re.compile(r"\s+")
_REPLACEMENT_CHARS_RE = re.compile("[�\x00]")

# Red-flag categories that map 1:1 to a C06 sub-criterion id. "other" maps
# to no sub-criterion.
_RED_FLAG_CATEGORIES = {
    "aml", "sanctions", "fraud", "market_abuse",
    "restricted_product", "suspicious_instruction",
}


def _clean_rule_text(text: str) -> str:
    """Collapse whitespace and strip obvious PDF-extraction noise."""
    text = _REPLACEMENT_CHARS_RE.sub("", text or "")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:MAX_RULE_CHARS]


def _prepared_rules(item: ChecklistItem) -> list[dict]:
    """Rules for this item (`retrieval.get_rules`: the prebuilt cache, or a
    live search for caller-added/edited items), deduped by rule_id (first
    seen kept), capped, and with cleaned/truncated text."""
    cap = MAX_RULES_LIST if item.answer_type == "list" else MAX_RULES
    seen: set[str] = set()
    out: list[dict] = []
    for rule in retrieval.get_rules(item):
        rule_id = rule.get("rule_id")
        if not rule_id or rule_id in seen:
            continue
        seen.add(rule_id)
        out.append({
            "rule_id": rule_id,
            "text": _clean_rule_text(rule.get("text", "")),
            "source": rule.get("source", ""),
            "page": rule.get("page"),
        })
        if len(out) >= cap:
            break
    return out


def _dedupe_refs(refs: list[EvidenceRef]) -> list[EvidenceRef]:
    seen: set[tuple[str, int, int, bool]] = set()
    out: list[EvidenceRef] = []
    for ref in refs:
        key = (ref.segment_id, ref.start_char, ref.end_char, ref.verified)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def _cannot_determine_result(
    item_id: str, sub_ids: list[str], reasoning: str, note: str,
    cfg_item: ChecklistItem | None = None,
) -> ItemResult:
    result = ItemResult(
        item_id=item_id,
        status=Status.CANNOT_DETERMINE,
        execution_status="prediction_failure",
        failure_category="analysis_unavailable",
        failure_stage="check_item",
        confidence=0.0,
        sub_criteria=[
            SubCriterionResult(id=sid, status=Status.CANNOT_DETERMINE, evidence=[], note=note)
            for sid in sub_ids
        ],
        evidence=[],
        rule_refs=[],
        reasoning=reasoning,
    )
    # Every sub-criterion is CANNOT_DETERMINE here, so any warn_if_unconfirmed
    # one gets a rule warning too -- the reviewer should still see it even
    # when the whole item failed to run (no segments, LLM failure, ...).
    apply_rule_warnings(result, cfg_item)
    return result


def validate_check_output(
    output: CheckItemOutput, item: ChecklistItem, segments: list[dict] | None = None,
) -> None:
    expected = {sc.id for sc in item.sub_criteria}
    actual = [sc.id for sc in output.sub_criteria]
    if len(actual) != len(expected) or set(actual) != expected:
        raise InvalidStructuredOutput("Every requested sub-criterion must be answered exactly once")
    if segments is not None:
        quotes = [quote for row in [*output.sub_criteria, *output.red_flags, *output.warnings]
                  for quote in row.evidence]
        if any(not quote.quote.strip() for quote in quotes):
            raise InvalidStructuredOutput("A provided evidence quote must contain original transcript words")
        _, diagnostics = resolve_evidence(segments, quotes)
        if diagnostics:
            raise InvalidStructuredOutput(
                "Evidence must cite an existing segment ID and words from that segment: "
                + diagnostics[0])


def allows_absence_review(item, review, segments, quality_flags, note) -> bool:
    """Accept a declared full-call absence finding, never silence about an obligation.

    This is a semantic answer from the normal check, not a metadata lookup,
    a default requirement type, or a reason to retry a valid prediction.
    """
    return bool(
        item.requirement_type != "OBLIGATION"
        and review is not None
        and review.prohibited_conduct.strip()
        and review.entire_transcript_reviewed
        and review.context_sufficient
        and note.strip()
        and any((seg.get("text") or "").strip() for seg in segments)
        and not ({"empty_transcript", "low_confidence", "single_speaker"} & set(quality_flags))
    )


def check_item_node(task: CheckItemTask) -> dict:
    item_id = task.get("item_id", "")
    # The run's own item config (caller checklist) when the fan-out sent it;
    # the default checklist otherwise.
    raw_item = task.get("checklist_item")
    item = ChecklistItem.model_validate(raw_item) if raw_item else config.get_item(item_id)

    if item is None:
        result = ItemResult(
            item_id=item_id,
            status=Status.CANNOT_DETERMINE,
            confidence=0.0,
            reasoning="Unknown checklist item id.",
        )
        return {
            "item_results": [result],
            "red_flags": [],
            "errors": [f"check_item[{item_id}]: unknown item id"],
        }

    sub_ids = [sc.id for sc in item.sub_criteria]
    segments = task.get("segments") or []
    errors: list[str] = []

    if not segments:
        return {"item_results": [_cannot_determine_result(
            item.id, sub_ids, "No transcript segments available for this call.",
            "No transcript available.", item)], "red_flags": []}

    try:
        rules = _prepared_rules(item)
    except Exception as exc:  # noqa: BLE001 - live corpus lookup failed; judge without rules
        rules = []
        errors.append(f"check_item[{item.id}]: rule lookup failed, judged without rules: "
                      f"{type(exc).__name__}: {exc}")

    transcript = format_transcript(segments)
    call_intent = task.get("call_intent")
    quality_flags = task.get("quality_flags") or []

    llm_result = get_llm_response(
        "check_item",
        SYSTEM,
        build_user(
            item.question,
            [{"id": sc.id, "text": sc.text, "absence": sc.absence} for sc in item.sub_criteria],
            item.answer_type,
            rules,
            transcript,
            call_intent,
        ),
        CheckItemOutput,
        validate=lambda output: validate_check_output(output, item, segments),
    )

    if not llm_result.ok:
        result = _cannot_determine_result(
            item.id, sub_ids,
            reasoning="Check could not be completed.",
            note="LLM output failed",
            cfg_item=item,
        )
        if llm_result.error == "InvalidStructuredOutput":
            result.failure_category = "incomplete_structured_output"
        return {
            "item_results": [result],
            "red_flags": [],
            "errors": errors + [f"check_item[{item.id}]: {llm_result.error}"],
        }

    return resolve_check_output(item, llm_result.value, segments, rules, quality_flags, errors)


def resolve_check_output(item, output, segments, rules, quality_flags, errors=None) -> dict:
    """Apply the same evidence and rollup rules to individual and grouped answers."""
    errors = list(errors or [])
    sub_ids = [sc.id for sc in item.sub_criteria]
    is_list = item.answer_type == "list"
    given_rule_ids = {rule["rule_id"] for rule in rules}

    # ---- Sub-criteria, matched by id in config order. ----------------
    answers_by_id: dict[str, object] = {}
    for ans in output.sub_criteria:
        if ans.id not in {sc.id for sc in item.sub_criteria}:
            if ans.id:
                errors.append(f"check_item[{item.id}]: unknown sub-criterion id {ans.id!r} ignored")
            continue
        answers_by_id[ans.id] = ans

    sub_results: dict[str, SubCriterionResult] = {}
    for sc in item.sub_criteria:
        ans = answers_by_id.get(sc.id)
        if ans is None:
            sub_results[sc.id] = SubCriterionResult(
                id=sc.id, status=Status.CANNOT_DETERMINE, evidence=[], note="not answered",
            )
            continue

        refs, notes = resolve_evidence(segments, ans.evidence)
        errors.extend(f"check_item[{item.id}]: {note}" for note in notes)

        try:
            status = Status(ans.status)
        except ValueError:
            errors.append(f"check_item[{item.id}]: invalid sub-criterion status {ans.status!r}")
            status = Status.CANNOT_DETERMINE

        note = clean_note(ans.note)

        absence_review = ans.absence_review
        reviewed_absence = allows_absence_review(item, absence_review, segments, quality_flags, note)
        configured_absence = sc.absence and absence_review is None
        if status == Status.PASS and not refs and not (configured_absence or reviewed_absence):
            status = Status.CANNOT_DETERMINE
            note = ("Absence finding lacks a complete, usable transcript review."
                    if absence_review is not None else "PASS without evidence")
        elif status == Status.FAIL and not (note or "").strip():
            note = "No supporting statement found in the call."

        sub_results[sc.id] = SubCriterionResult(
            id=sc.id, status=status, evidence=refs, note=note,
            absence_review=absence_review if reviewed_absence and status == Status.PASS else None,
        )

    # ---- Red flags (list-type items only). ----------------------------
    red_flags: list[RedFlag] = []
    if is_list:
        for rf in output.red_flags:
            refs, notes = resolve_evidence(segments, rf.evidence)
            errors.extend(f"check_item[{item.id}]: {note}" for note in notes)
            if not refs:
                errors.append(
                    f"check_item[{item.id}]: red flag {rf.category!r} dropped, no evidence resolved"
                )
                continue
            red_flags.append(RedFlag(
                category=rf.category, severity=rf.severity, description=rf.description, evidence=refs,
            ))

        for flag in red_flags:
            if flag.category not in _RED_FLAG_CATEGORIES:
                continue
            target = sub_results.get(flag.category)
            if target is None:
                continue
            merged_note = f"{target.note} {flag.description}".strip() if target.note else flag.description
            merged_evidence = _dedupe_refs(list(target.evidence) + list(flag.evidence))
            sub_results[flag.category] = SubCriterionResult(
                id=target.id, status=Status.FAIL, evidence=merged_evidence, note=merged_note,
            )

    # ---- Warnings: possible/minor issues for a human reviewer, resolved --
    # like evidence quotes but never changing any status. ------------------
    warnings: list[ReviewWarning] = []
    for w in output.warnings:
        message = clean_note(w.message)
        if not message:
            continue
        sub_id = w.sub_criterion_id
        if sub_id is not None and sub_id not in sub_ids:
            errors.append(
                f"check_item[{item.id}]: warning cites unknown sub_criterion_id {sub_id!r}; set to None"
            )
            sub_id = None
        refs, notes = resolve_evidence(segments, w.evidence)
        errors.extend(f"check_item[{item.id}]: {note}" for note in notes)
        warnings.append(ReviewWarning(
            sub_criterion_id=sub_id,
            severity=w.severity,
            message=message,
            evidence=refs,
            source="llm",
        ))

    # ---- Item status rollup (never the LLM's). -------------------------
    ordered_results = [sub_results[sc.id] for sc in item.sub_criteria]
    item_status = rollup_status(ordered_results)

    item_evidence: list[EvidenceRef] = []
    for r in ordered_results:
        item_evidence.extend(r.evidence)
    item_evidence = _dedupe_refs(item_evidence)

    rule_refs = [r for r in output.rule_refs if r in given_rule_ids]

    confidence = max(0.0, min(1.0, output.confidence))
    reasoning = output.reasoning
    if DEGRADING_FLAGS & set(quality_flags):
        if confidence > DEGRADED_CONFIDENCE_CAP:
            confidence = DEGRADED_CONFIDENCE_CAP
        reasoning = (
            f"{reasoning} (confidence capped: transcript quality flags "
            f"{sorted(DEGRADING_FLAGS & set(quality_flags))})"
        ).strip()

    result = ItemResult(
        item_id=item.id,
        status=item_status,
        confidence=confidence,
        sub_criteria=ordered_results,
        evidence=item_evidence,
        rule_refs=rule_refs,
        reasoning=reasoning,
        warnings=warnings,
    )
    if any(sc.id not in answers_by_id for sc in item.sub_criteria):
        result.execution_status = "prediction_failure"
        result.failure_category = "incomplete_structured_output"
        result.failure_stage = "check_item"
        errors.append(f"check_item[{item.id}]: missing sub-criterion answer")
    apply_rule_warnings(result, item)

    out: dict = {"item_results": [result], "red_flags": red_flags}
    if errors:
        out["errors"] = errors
    return out
