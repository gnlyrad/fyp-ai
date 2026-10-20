"""Share transcript input across a bounded group; repair only unusable items."""
from __future__ import annotations

import os
from collections import Counter

from compliance_pipeline.llm import get_llm_response
from compliance_pipeline.nodes._common import format_transcript
from compliance_pipeline.nodes.check_item import (
    _prepared_rules, check_item_node, resolve_check_output, validate_check_output,
)
from compliance_pipeline.prompts.check_item import SYSTEM, build_batch_user
from compliance_pipeline.schemas import CheckBatchOutput, ChecklistItem


def check_batch_size() -> int:
    value = os.getenv("COMPLIANCE_CHECK_BATCH_SIZE", "1")
    if value not in {str(n) for n in range(1, 21)}:
        raise ValueError("COMPLIANCE_CHECK_BATCH_SIZE must be an integer from 1 to 20")
    return int(value)


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
    answers = {}
    if segments:
        result = get_llm_response(
            "check_batch", SYSTEM.replace("ONE checklist item at a time", "each checklist item independently"),
            build_batch_user(items, rules, format_transcript(segments), task.get("call_intent")),
            CheckBatchOutput,
        )
        if result.ok:
            counts = Counter(answer.item_id for answer in result.value.answers)
            answers = {answer.item_id: answer.result for answer in result.value.answers
                       if counts[answer.item_id] == 1}
    combined = {"item_results": [], "red_flags": [], "errors": []}
    for item in items:
        answer = answers.get(item.id)
        try:
            if answer is None:
                raise ValueError("Missing or duplicate grouped answer")
            validate_check_output(answer, item, segments)
        except ValueError:
            # Successful siblings are retained; malformed/missing rows get the existing
            # single-item recovery path, which never forces a semantic Unclear result.
            update = check_item_node({**task, "item_id": item.id, "checklist_item": item.model_dump()})
        else:
            update = resolve_check_output(item, answer, segments, rules[item.id],
                                          task.get("quality_flags") or [], diagnostics[item.id])
        for key in combined:
            combined[key].extend(update.get(key, []))
    return combined
