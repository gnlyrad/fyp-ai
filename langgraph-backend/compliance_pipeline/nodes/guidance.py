"""Guidance node: fetch the regulatory standard each checklist item is judged by.

The analysis prompt used to ask "were fees explained?" without ever saying what
counts as explained, leaving the model to supply its own bar. This node looks the
bar up in the MAS corpus so the judgement is made against a stated standard, and
so the finding can cite the notice it rests on.

Only `targeted` and `exhaustive` items are given guidance:

  targeted      - adequacy judgements ("was suitability *established*"), which is
                  exactly the question regulation answers
  exhaustive    - prohibited-conduct sweeps ("undue urgency"), where the corpus
                  supplies the catalogue of what counts
  deterministic - deliberately excluded. A presence check is decided by whether
                  the line is in the excerpts; its prompt turns silence into Fail,
                  and handing the model regulatory prose gives it material to
                  argue its way out of that Fail.
  audio_metrics - no model call at all, so nothing to ground.

Batched over the checklist in one pass, like `retrieval_node`.
"""
from __future__ import annotations

from typing import Any

from ..knowledge import search_regulations
from ..state import DEFAULT_QUESTION_TYPE, ChecklistItem, ComplianceState

GUIDED_TYPES = {"targeted", "exhaustive"}


def guidance_query(item: ChecklistItem) -> str:
    """The question plus its retrieval terms.

    Both, because the question names the control and `evidence_query` carries the
    domain vocabulary ("annual management fee", "entry charge") that actually
    matches the regulatory text.
    """
    return " ".join(filter(None, [item["question"], item.get("evidence_query")]))


def guidance_node(state: ComplianceState) -> dict[str, Any]:
    """Regulatory extracts per guided checklist item; absent ids simply get none."""
    guidance = {}
    for item in state["checklist"]:
        question_type = item.get("question_type") or DEFAULT_QUESTION_TYPE
        if question_type not in GUIDED_TYPES:
            continue
        passages = search_regulations.invoke({"query": guidance_query(item)})
        if passages:
            guidance[item["id"]] = passages
    return {"guidance": guidance}
