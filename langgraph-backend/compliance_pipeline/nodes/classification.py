"""Classification node: label each checklist question with how it must be answered.

Sits between ingestion and retrieval, because the type decides how much of the
transcript retrieval hands over — an exhaustive question needs all of it, a
targeted or deterministic one needs only the relevant excerpts.

One LLM call labels the whole checklist. Items that already carry a
`question_type` are left alone, so the built-in checklist (which ships tagged)
costs nothing and tests can pin types without mocking the model.
"""
from __future__ import annotations

from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..model import get_model
from ..prompts.classification import CLASSIFICATION_SYSTEM_PROMPT
from ..state import (
    DEFAULT_QUESTION_TYPE,
    QUESTION_TYPES,
    ChecklistItem,
    ComplianceState,
)


class ItemType(BaseModel):
    """One classified checklist question."""

    id: str = Field(description="The checklist item id, echoed exactly as given.")
    question_type: str = Field(
        description="One of: deterministic, exhaustive, targeted"
    )
    rationale: str = Field(description="One sentence justifying the type.")
    evidence_query: str = Field(
        default="",
        description="Short keyword string for lexical retrieval.",
    )


class ChecklistClassification(BaseModel):
    """Schema the model is forced to fill in — one entry per submitted question."""

    items: list[ItemType] = Field(default_factory=list)


def normalise_question_type(value: str) -> str | None:
    """Map the model's answer onto our vocabulary; None when it says nothing usable."""
    text = (value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "rule_based": "deterministic",
        "binary": "deterministic",
        "presence": "deterministic",
        "comprehensive": "exhaustive",
        "specific": "targeted",
    }
    text = aliases.get(text, text)
    return text if text in QUESTION_TYPES else None


def _prompt(pending: list[ChecklistItem]) -> str:
    lines = [f'{item["id"]}: {item["question"]}' for item in pending]
    return "Classify each of these compliance questions:\n\n" + "\n".join(lines)


def classification_node(state: ComplianceState) -> dict[str, Any]:
    """Attach a `question_type` (and, where missing, an `evidence_query`) to every item."""
    checklist = list(state["checklist"])
    pending = [item for item in checklist if not item.get("question_type")]
    if not pending:
        return {"checklist": checklist}

    classifier = get_model().with_structured_output(ChecklistClassification)
    try:
        answer = classifier.invoke([
            SystemMessage(content=CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=_prompt(pending)),
        ])
        # Keyed by id rather than trusted positionally: the model may reorder,
        # drop, or invent entries, and a silently shifted type would reroute the
        # wrong question.
        classified = {entry.id: entry for entry in (answer.items or [])}
        error = None
    except Exception as exc:  # model unavailable, throttled, or malformed output
        classified, error = {}, exc

    enriched: list[ChecklistItem] = []
    for item in checklist:
        if item.get("question_type"):
            enriched.append(item)
            continue

        entry = classified.get(item["id"])
        question_type = normalise_question_type(entry.question_type) if entry else None
        if question_type is None:
            # Falling back to `targeted` reproduces the pre-classification
            # behaviour, which is the safe default: it retrieves rather than
            # assuming absence, and never skips the model.
            reason = (
                f"classification unavailable: {error}" if error is not None
                else "classification returned no usable type; defaulted to targeted"
            )
            enriched.append({**item, "question_type": DEFAULT_QUESTION_TYPE, "type_rationale": reason})
            continue

        classified_item: ChecklistItem = {
            **item,
            "question_type": question_type,
            "type_rationale": entry.rationale,
        }
        # Caller-supplied items carry no `evidence_query`, which left them
        # retrieving on the question wording alone. Backfill it here at no
        # extra cost; never overwrite one the caller did supply.
        if not item.get("evidence_query") and entry.evidence_query.strip():
            classified_item["evidence_query"] = entry.evidence_query.strip()
        enriched.append(classified_item)

    return {"checklist": enriched}
