"""Share transcript input across a bounded group; repair only unusable items."""
from __future__ import annotations

import os
import logging
from collections import Counter

from compliance_pipeline.llm import get_llm_response
from compliance_pipeline.nodes._common import format_transcript
from compliance_pipeline.nodes.check_item import (
    _prepared_rules, _cannot_determine_result, check_item_node, resolve_check_output, validate_check_output,
)
from compliance_pipeline.prompts.check_item import SYSTEM, build_batch_user
from compliance_pipeline.schemas import CheckBatchOutput, ChecklistItem


logger = logging.getLogger(__name__)


def check_batch_size() -> int:
    value = os.getenv("COMPLIANCE_CHECK_BATCH_SIZE", "1")
    if value not in {str(n) for n in range(1, 21)}:
        raise ValueError("COMPLIANCE_CHECK_BATCH_SIZE must be an integer from 1 to 20")
    return int(value)


def check_recovery_batch_size() -> int:
    """Opt-in grouping of unusable output only; normal semantic answers are final."""
    value = os.getenv("COMPLIANCE_CHECK_RECOVERY_BATCH_SIZE", "1")
    if value not in {str(n) for n in range(1, 21)}:
        raise ValueError("COMPLIANCE_CHECK_RECOVERY_BATCH_SIZE must be an integer from 1 to 20")
    return int(value)


def _group_request(items, rules, segments, call_intent):
    result = get_llm_response(
        "check_batch", SYSTEM.replace("ONE checklist item at a time", "each checklist item independently"),
        build_batch_user(items, rules, format_transcript(segments), call_intent),
        CheckBatchOutput,
    )
    counts = Counter(answer.item_id for answer in result.value.answers) if result.ok else {}
    answers = {answer.item_id: answer.result for answer in result.value.answers
               if counts[answer.item_id] == 1} if result.ok else {}
    valid, issues = {}, Counter()
    for item in items:
        answer = answers.get(item.id)
        if answer is None:
            issues["missing_or_duplicate" if result.ok else "group_unavailable"] += 1
            continue
        try:
            validate_check_output(answer, item, segments)
        except ValueError:
            issues["invalid_structure_or_evidence"] += 1
        else:
            valid[item.id] = answer
    if issues:
        # Counts only: never persist returned quotes or transcript text in diagnostics.
        logger.warning("check_batch unusable output counts: %s", dict(issues))
    return result, valid


def _unavailable(item, error, diagnostics):
    native = os.getenv("COMPLIANCE_CHECK_TRANSPORT", "standard") == "batch"
    explanation = ("Native batch unavailable; inspect job receipts before resubmitting."
                   if native else "Check unavailable due to a non-retryable provider error.")
    return {
        "item_results": [_cannot_determine_result(
            item.id, [sub.id for sub in item.sub_criteria], explanation,
            "LLM output unavailable.", item)],
        "red_flags": [],
        "errors": diagnostics + [f"check_item[{item.id}]: {error}"],
    }


def check_batch_node(task: dict) -> dict:
    items = [ChecklistItem.model_validate(raw) for raw in task["checklist_items"]]
    segments = task.get("segments") or []
    rules, diagnostics = {}, {}
    for item in items:
        try:
            rules[item.id] = _prepared_rules(item) if segments else []
            diagnostics[item.id] = []
        except Exception as exc:
            rules[item.id] = []
            diagnostics[item.id] = [f"check_item[{item.id}]: rule lookup failed: {type(exc).__name__}"]
    recovery_size = check_recovery_batch_size()
    answers, blocked = {}, {}
    if segments:
        result, answers = _group_request(items, rules, segments, task.get("call_intent"))
        if not result.ok and not result.recovery_allowed:
            blocked = {item.id: result.error for item in items}
        else:
            pending = [item for item in items if item.id not in answers]
            # Exactly one additional grouped recovery pass. Retain successful siblings;
            # never include a structurally valid PASS, FAIL or semantic unclear answer.
            if recovery_size > 1:
                for offset in range(0, len(pending), recovery_size):
                    group = pending[offset:offset + recovery_size]
                    if len(group) < 2:
                        continue
                    repaired, usable = _group_request(group, rules, segments, task.get("call_intent"))
                    answers.update(usable)
                    if not repaired.ok and not repaired.recovery_allowed:
                        blocked.update({item.id: repaired.error for item in group})
    combined = {"item_results": [], "red_flags": [], "errors": []}
    for item in items:
        if item.id in blocked:
            # A permanent provider error or pending/uncertain native job cannot be
            # submitted again through recovery or individual fallback.
            update = _unavailable(item, blocked[item.id], diagnostics[item.id])
        elif item.id in answers:
            update = resolve_check_output(item, answers[item.id], segments, rules[item.id],
                                          task.get("quality_flags") or [], diagnostics[item.id])
        else:
            # Retain the bounded existing per-item path for remaining unusable rows.
            update = check_item_node({**task, "item_id": item.id, "checklist_item": item.model_dump()})
        for key in combined:
            combined[key].extend(update.get(key, []))
    return combined
