"""Pydantic models for the compliance pipeline (CLAUDE.md "Schemas" section).

Two families live here:
- Pipeline models (`ItemResult`, `RedFlag`, `Order`, `Action`, ...) that flow
  through `ComplianceState` and get written to the output.
- LLM-output models (`ExtractedOrder`, `ProposedAction`) that omit fields the
  pipeline fills in code (`product_id`, `owner_role`), so the LLM cannot set
  them itself.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Status(str, Enum):
    """The one status enum used everywhere. CANNOT_DETERMINE is a valid answer."""

    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    CANNOT_DETERMINE = "CANNOT_DETERMINE"


class SubCriterionResult(BaseModel):
    id: str
    status: Status
    evidence_ids: list[str] = Field(default_factory=list)
    note: str = ""


class ItemResult(BaseModel):
    item_id: str
    status: Status
    confidence: float = 0.0  # 0 to 1
    sub_criteria: list[SubCriterionResult] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    rule_refs: list[str] = Field(default_factory=list)  # IDs of rule chunks relied on
    reasoning: str = ""  # short, for reviewers
    verified: bool = False


class RedFlag(BaseModel):
    category: Literal[
        "aml", "sanctions", "fraud", "market_abuse",
        "restricted_product", "suspicious_instruction", "other",
    ]
    severity: Literal["low", "medium", "high"]
    description: str
    evidence_ids: list[str] = Field(default_factory=list)


class Order(BaseModel):
    product_name: str | None = None
    product_id: str | None = None  # filled by normalise.py
    direction: Literal["buy", "sell"] | None = None
    quantity: float | None = None
    amount: float | None = None
    currency: str | None = None  # ISO 4217
    order_type: str | None = None  # market, limit, stop, and so on
    timing: str | None = None
    confirmed: bool = False  # explicitly confirmed by the client
    superseded: bool = False  # replaced by a later correction
    evidence_ids: list[str] = Field(default_factory=list)


class Action(BaseModel):
    action_type: str  # key in ownership.yaml
    description: str
    owner_role: str | None = None  # filled by code
    evidence_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM-output variants: same shape minus the pipeline-filled field(s).
# ---------------------------------------------------------------------------

class ExtractedOrder(BaseModel):
    """What `extraction` asks the LLM for. `normalise.py` fills `product_id`."""

    product_name: str | None = None
    direction: Literal["buy", "sell"] | None = None
    quantity: float | None = None
    amount: float | None = None
    currency: str | None = None
    order_type: str | None = None
    timing: str | None = None
    confirmed: bool = False
    superseded: bool = False
    evidence_ids: list[str] = Field(default_factory=list)


class ProposedAction(BaseModel):
    """What `decide` asks the LLM for. Code fills `owner_role` from ownership.yaml."""

    action_type: str
    description: str
    evidence_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Supporting models: transcript segments, call intent, escalation, config.
# ---------------------------------------------------------------------------

class Segment(BaseModel):
    """One turn of the transcript, produced by `nodes/ingest.py`.

    A "turn" is a maximal run of consecutive speech by the same speaker;
    `turn` numbers turns 1..N across the whole call. `id` (`seg_0001`, ...)
    is the stable evidence id used everywhere downstream.
    """

    id: str
    speaker: str
    turn: int
    start: float
    end: float
    text: str
    role: str | None = None
    confidence: float | None = None
    word_count: int = 0


class CallIntent(BaseModel):
    category: Literal["enquiry", "execution_only", "advisory"]
    confidence: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)


class Escalation(BaseModel):
    required: bool = False
    reasons: list[str] = Field(default_factory=list)
    hold_transaction: bool = False


class SubCriterion(BaseModel):
    """One row of a checklist item's `sub_criteria` config."""

    id: str
    text: str


class ChecklistItem(BaseModel):
    """The shape of one entry in `config/checklist.yaml` (CLAUDE.md "Checklist config")."""

    id: str
    name: str
    stage: Literal[1, 2, 3]
    answer_type: Literal["yes_no", "category", "criteria", "entities", "list"]
    applies_when: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    needs_data: list[str] = Field(default_factory=list)
    rule_queries: list[str] = Field(default_factory=list)
    question: str = ""
    sub_criteria: list[SubCriterion] = Field(default_factory=list)
