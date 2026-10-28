"""Pydantic models for the compliance pipeline (CLAUDE.md "Schemas" section).

Two families live here:
- Pipeline models (`ItemResult`, `RedFlag`, `Action`, ...) that flow through
  `ComplianceState` and get written to the output. These persist evidence as
  `EvidenceRef` (segment id + character span), never quoted text.
- LLM-output models (`ProposedAction`, `IntentObservation`, ...) that cite
  evidence as a short verbatim `EvidenceQuote` instead, since LLMs are poor
  at counting characters. `nodes/_common.py:resolve_evidence` turns a list of
  quotes into a list of resolved `EvidenceRef`s; quotes themselves are never
  persisted.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Status(str, Enum):
    """The one status enum used everywhere. CANNOT_DETERMINE is a valid answer."""

    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    CANNOT_DETERMINE = "CANNOT_DETERMINE"


# ---------------------------------------------------------------------------
# Evidence: refs are persisted, quotes are LLM-output-only and never stored.
# ---------------------------------------------------------------------------

class EvidenceRef(BaseModel):
    """A resolved evidence pointer: a character span into one segment's
    `text` (offsets are relative to that segment's own text, which is
    immutable after ingest -- unlike the prefixed transcript string, its
    offsets stay stable regardless of how the transcript is formatted)."""

    segment_id: str
    start_char: int
    end_char: int
    verified: bool = False


class EvidenceQuote(BaseModel):
    """What the LLM cites: a short verbatim quote from one segment. Resolved
    to an `EvidenceRef` in code by `nodes/_common.py:resolve_evidence`;
    quotes are never persisted."""

    segment_id: str
    quote: str = ""


class AbsenceReview(BaseModel):
    """An explicit transcript-wide finding, not inferred checklist metadata.

    Absence has no spoken quote. Record what was searched for and whether
    the available context supports that finding instead of fabricating one.
    """

    prohibited_conduct: str = Field(description="The conduct the criterion prohibits; never a required act.")
    entire_transcript_reviewed: bool = False
    context_sufficient: bool = False


class SubCriterionResult(BaseModel):
    id: str
    status: Status
    evidence: list[EvidenceRef] = Field(default_factory=list)
    note: str = ""
    absence_review: AbsenceReview | None = None


class ReviewWarning(BaseModel):
    """A possible or minor issue for a human reviewer to look at -- never
    enough to change a status or the rollup. Named `ReviewWarning` to avoid
    shadowing the builtin `Warning`. Every result goes to human evaluation,
    so it is safer to over-report than to omit (CLAUDE.md "Warnings")."""

    sub_criterion_id: str | None = None
    severity: Literal["low", "medium"] = "low"
    message: str = ""
    evidence: list[EvidenceRef] = Field(default_factory=list)
    source: Literal["llm", "rule"] = "llm"


class ItemResult(BaseModel):
    item_id: str
    status: Status
    execution_status: Literal["success", "prediction_failure"] = "success"
    failure_category: str | None = None
    failure_stage: str | None = None
    confidence: float = 0.0  # 0 to 1
    sub_criteria: list[SubCriterionResult] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    rule_refs: list[str] = Field(default_factory=list)  # IDs of rule chunks relied on
    reasoning: str = ""  # short, for reviewers
    verified: bool = False
    warnings: list[ReviewWarning] = Field(default_factory=list)


class RedFlag(BaseModel):
    category: Literal[
        "aml", "sanctions", "fraud", "market_abuse",
        "restricted_product", "suspicious_instruction", "other",
    ]
    severity: Literal["low", "medium", "high"]
    description: str
    evidence: list[EvidenceRef] = Field(default_factory=list)


class Action(BaseModel):
    action_type: str  # key in ownership.yaml
    description: str
    owner_role: str | None = None  # filled by code
    evidence: list[EvidenceRef] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LLM-output variants: cite EvidenceQuote, never EvidenceRef.
# ---------------------------------------------------------------------------

class ProposedAction(BaseModel):
    """What `decide` asks the LLM for. Code fills `owner_role` from ownership.yaml
    and resolves `evidence` quotes to refs."""

    action_type: str
    description: str
    evidence: list[EvidenceQuote] = Field(default_factory=list)


class SpeakerRoleGuess(BaseModel):
    """What `role_inference` asks the LLM for, one entry per speaker label.
    Not persisted, so its evidence quotes are used as-is (no resolution)."""

    speaker: str
    role: Literal["Advisor", "Client", "uncertain"]
    confidence: float = 0.0  # 0 to 1
    reason: str = ""
    evidence: list[EvidenceQuote] = Field(default_factory=list)


class RoleInferenceOutput(BaseModel):
    speakers: list[SpeakerRoleGuess] = Field(default_factory=list)


class IntentObservation(BaseModel):
    """One intent the LLM says was reached during the call, with evidence.

    `call_intent` asks for every intent reached rather than a single final
    label; code (not the LLM) picks the most regulated one to report."""

    category: Literal["enquiry", "execution_only", "advisory"]
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    reason: str = ""


class CallIntentOutput(BaseModel):
    """What `call_intent` asks the LLM for. Code derives the final category."""

    observations: list[IntentObservation] = Field(default_factory=list)
    confidence: float = 0.0  # 0 to 1: how sure this is a classifiable financial-services call
    not_financial_call: bool = False
    reasoning: str = ""


class SubCriterionAnswer(BaseModel):
    """What `check_item` asks the LLM for, one entry per configured
    sub-criterion id. Code (not the LLM) rolls these up into the item's
    overall status and resolves `evidence` quotes to refs."""

    id: str = Field("", description="The sub-criterion id exactly as given.")
    # NOT_APPLICABLE is not offered to the LLM: only the gate (applies_when) sets it.
    status: Literal["PASS", "FAIL", "CANNOT_DETERMINE"]
    evidence: list[EvidenceQuote] = Field(default_factory=list)
    absence_review: AbsenceReview | None = Field(None, description=(
        "Only for PASS based on no prohibited conduct being observed after full transcript review. "
        "Never for a missing required disclosure, check, consent or confirmation. Otherwise null."
    ))
    note: str = Field("", description=(
        "One or two plain sentences explaining the judgement. "
        "No status, id, or field names."
    ))

    @field_validator("status", mode="before")
    @classmethod
    def _not_applicable_is_cannot_determine(cls, value):
        """A stray N/A from the model is read as "cannot determine", not rejected."""
        if isinstance(value, str) and value.strip().upper().replace("/", "_").replace(" ", "_") in {"NOT_APPLICABLE", "N_A"}:
            return "CANNOT_DETERMINE"
        return value


class RedFlagAnswer(BaseModel):
    """What `check_item` asks the LLM for on `answer_type: list` items
    (C06). Resolved in code into a persisted `RedFlag`."""

    category: Literal[
        "aml", "sanctions", "fraud", "market_abuse",
        "restricted_product", "suspicious_instruction", "other",
    ] = "other"
    severity: Literal["low", "medium", "high"] = "low"
    description: str = ""
    evidence: list[EvidenceQuote] = Field(default_factory=list)


class WarningAnswer(BaseModel):
    """What `check_item` asks the LLM for alongside `sub_criteria`: possible
    or minor issues a human reviewer should look at even when they don't
    make a sub-criterion FAIL. Resolved in code into a persisted
    `ReviewWarning`; never changes any status."""

    sub_criterion_id: str | None = Field(None, description=(
        "The sub-criterion id this warning relates to, exactly as given, "
        "or omitted/null if it doesn't tie to one specific sub-criterion."
    ))
    severity: Literal["low", "medium"] = "low"
    message: str = ""
    evidence: list[EvidenceQuote] = Field(default_factory=list)


class CheckItemOutput(BaseModel):
    """What `check_item` asks the LLM for. Code derives the item's overall
    `Status` from `sub_criteria` rather than trusting the LLM's rollup."""

    sub_criteria: list[SubCriterionAnswer] = Field(default_factory=list)
    red_flags: list[RedFlagAnswer] = Field(default_factory=list)
    rule_refs: list[str] = Field(default_factory=list, description=(
        "Ids of the regulatory extracts that set the standard applied, copied "
        "from the [rule <id>] headers."
    ))
    confidence: float = 0.0  # 0 to 1
    reasoning: str = Field("", description="At most 3 sentences.")
    warnings: list[WarningAnswer] = Field(default_factory=list, description=(
        "Possible or minor issues a human reviewer should look at even when "
        "they don't make a sub-criterion FAIL."
    ))


class GroupedItemAnswer(BaseModel):
    item_id: str
    result: CheckItemOutput


class CheckBatchOutput(BaseModel):
    answers: list[GroupedItemAnswer]


class VerifyVerdict(BaseModel):
    """One verdict from `verify`'s second-reviewer LLM call, one per target
    (a sub-criterion or a red flag) it was asked to check."""

    target_id: str = ""
    verdict: Literal["upheld", "not_supported"]
    reason: str = ""
    counter_evidence: list[EvidenceQuote] = Field(default_factory=list)


class VerifyOutput(BaseModel):
    """What `verify` asks the LLM for: a verdict for every target listed
    in the prompt."""

    verdicts: list[VerifyVerdict] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Supporting models: transcript segments, call intent, escalation, config.
# ---------------------------------------------------------------------------

class Segment(BaseModel):
    """One turn of the transcript, produced by `nodes/ingest.py`.

    A "turn" is a maximal run of consecutive speech by the same speaker;
    `turn` numbers turns 1..N across the whole call. `id` (`seg_0001`, ...)
    is the stable evidence id used everywhere downstream, and `text` is
    immutable after ingest -- evidence offsets are relative to it.
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
    role_source: str | None = None  # "caller" | "channel" | "llm" | None


class CallIntent(BaseModel):
    """`nodes/call_intent.py`'s output. `category` is None when the LLM
    failed, the call wasn't a classifiable financial-services call, no
    observation survived evidence resolution, or confidence was too low.
    `reached` lists every intent category reached, ranked most- to
    least-regulated (advisory > execution_only > enquiry)."""

    category: Literal["enquiry", "execution_only", "advisory"] | None = None
    confidence: float = 0.0
    evidence: list[EvidenceRef] = Field(default_factory=list)
    reached: list[str] = Field(default_factory=list)
    reasoning: str = ""


class Escalation(BaseModel):
    required: bool = False
    reasons: list[str] = Field(default_factory=list)
    hold_transaction: bool = False
    source: Literal["rules", "llm", "none", "undetermined"] = "none"
    rule_ids: list[str] = Field(default_factory=list)


class EscalationDecision(BaseModel):
    """What `decide` asks the LLM for when no `escalation_rules.yaml` rule
    fires but some Stage 2 result is PARTIAL or CANNOT_DETERMINE."""

    escalate: bool = False
    hold: bool = False
    reason: str = ""


class SubCriterion(BaseModel):
    """One row of a checklist item's `sub_criteria` config.

    `absence: true` marks a "nothing indicates X" check, which can PASS with no
    evidence (there is nothing to quote). Every other PASS needs evidence.
    """

    id: str
    text: str
    absence: bool = False
    warn_if_unconfirmed: bool = False


class ChecklistItem(BaseModel):
    """The shape of one entry in `config/checklist.yaml` (CLAUDE.md "Checklist config")."""

    id: str
    name: str
    stage: Literal[1, 2, 3]
    answer_type: Literal["yes_no", "category", "criteria", "entities", "list"]
    applies_when: dict = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    rule_queries: list[str] = Field(default_factory=list)
    question: str = ""
    requirement_type: Literal["OBLIGATION", "PROHIBITION"] | None = None
    sub_criteria: list[SubCriterion] = Field(default_factory=list)
