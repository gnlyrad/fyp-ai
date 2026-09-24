"""`check_item`: evaluate one checklist item (one `Send` branch per applicable item).

Final intended behaviour (CLAUDE.md): load the item's config, its cached
rules (`retrieval.get_rules`), the transcript, and only the context keys
listed in `needs_data`; call the LLM via `llm.get_llm_response`; derive the
item's overall status from its sub-criteria in code (all PASS -> PASS, all
FAIL -> FAIL, mix -> PARTIAL, any CANNOT_DETERMINE with no FAIL ->
CANNOT_DETERMINE) rather than trusting the LLM's rollup. If a required
context key is missing, return CANNOT_DETERMINE without calling the LLM.
C06 (`answer_type: list`) additionally returns `RedFlag`s. Evidence ids not
present in the segment set are dropped in code with an error note.

Step-0 stub: always returns a single CANNOT_DETERMINE result for the item.
"""
from __future__ import annotations

from compliance.schemas import ItemResult, Status
from compliance.state import CheckItemTask


def check_item_node(task: CheckItemTask) -> dict:
    result = ItemResult(
        item_id=task["item_id"],
        status=Status.CANNOT_DETERMINE,
        confidence=0.0,
        sub_criteria=[],
        evidence_ids=[],
        rule_refs=[],
        reasoning="not implemented",
    )
    return {"item_results": [result]}
