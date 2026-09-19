"""Graph state for the compliance pipeline.

`findings` and `spans` use `operator.add` so that the fanned-out analysis nodes
can each return a one-element list and have LangGraph concatenate them, rather
than the last writer overwriting the rest.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Verdict = Literal["Pass", "Fail", "Unclear", "N/A"]
SpanLabel = Literal["compliant", "violation"]

# How a checklist question has to be answered, which decides both how much of the
# transcript retrieval hands over and which analysis node judges it.
#   deterministic - fixed rule/phrase presence; absence is a Fail, not an Unclear
#   exhaustive    - every occurrence matters, so the whole transcript is needed
#   targeted      - one specific passage, found by retrieval (the default)
QuestionType = Literal["deterministic", "exhaustive", "targeted"]
QUESTION_TYPES: tuple[QuestionType, ...] = ("deterministic", "exhaustive", "targeted")
DEFAULT_QUESTION_TYPE: QuestionType = "targeted"


class Chunk(TypedDict, total=False):
    chunk_id: str
    text: str
    # Offsets into the transcript. Always present.
    start_char: int
    end_char: int
    # Present only when the chunk came from a diarized turn (AWS Transcribe
    # `audio_segments`) rather than sentence-splitting a plain string.
    start_seconds: float
    end_seconds: float
    speaker_label: str  # e.g. "spk_0", as Transcribe assigned it


class AudioSegment(TypedDict, total=False):
    """One diarized turn, as AWS Transcribe's `results.audio_segments` shapes it."""

    transcript: str
    start_time: str
    end_time: str
    speaker_label: str


class RetrievedChunk(Chunk, total=False):
    score: float


class ChecklistItem(TypedDict, total=False):
    id: str
    question: str
    evidence_query: str
    # Filled by the classification node when the caller omits it. Supplying it in
    # the request skips classification entirely for that item.
    question_type: QuestionType
    type_rationale: str  # one line, so the routing decision stays auditable


class RegulationPassage(TypedDict, total=False):
    """An extract from the MAS corpus, carrying enough to cite it.

    This is the *standard* a checklist item is judged against, never evidence of
    what happened on the call. It can therefore never become a `Span`.
    """

    text: str
    source: str  # pdf filename, relative to the corpus folder
    page: int  # 1-based, as printed
    score: float


class Span(TypedDict, total=False):
    """A verbatim substring of the transcript, located by character offset.

    `label` is reconciled with the finding's verdict; `model_label` preserves what
    the model originally said so a disagreement stays auditable rather than silent.
    """

    text: str
    start: int
    end: int
    label: SpanLabel
    model_label: SpanLabel | None
    label_corrected: bool
    checklist_id: str
    chunk_id: str | None
    # The raw speaker id (e.g. "spk_0") the source chunk carried, for a caller
    # that wants to display evidence by speaker. Absent for a plain-text
    # transcript (not diarized turns).
    speaker_label: str | None
    # The source chunk's real timing, carried through for a caller that wants to
    # seek audio playback to this passage. Absent for a plain-text transcript.
    start_seconds: float | None
    end_seconds: float | None


class Finding(TypedDict, total=False):
    checklist_id: str
    question: str
    question_type: QuestionType
    verdict: Verdict
    reason: str
    confidence: Literal["high", "low"]
    spans: list[Span]
    # Convenience view of the first verified span.
    evidence_quote: str | None
    chunk_id: str | None
    retrieved_evidence: list[RetrievedChunk]
    # The regulatory extracts this judgement was made under. Empty for question
    # types that are not given guidance, and when the corpus is unreachable.
    regulatory_basis: list[RegulationPassage]


class AnalyseInput(TypedDict, total=False):
    """The public API body — what `POST /runs/wait` accepts as `input`.

    Supply either `transcript` (a paragraph or a whole call, as plain text) or
    `audio_segments` (AWS Transcribe's `results.audio_segments`, diarized and
    timestamped) — not both; `audio_segments` takes priority if both are given.
    `checklist` runs against whichever is supplied; omitting it falls back to
    the built-in one.
    """

    transcript: str
    audio_segments: list[AudioSegment]
    checklist: list[ChecklistItem]


class ComplianceState(TypedDict, total=False):
    """State for the top-level graph."""

    # --- ingestion ---
    transcript: str  # the text every span offset refers to
    audio_segments: list[AudioSegment]  # raw input, if the caller supplied it
    chunks: list[Chunk]
    checklist: list[ChecklistItem]

    # --- guidance ---
    guidance: dict[str, list[RegulationPassage]]  # checklist_id -> regulatory extracts

    # --- retrieval ---
    evidence: dict[str, list[RetrievedChunk]]  # checklist_id -> retrieved chunks

    # --- analysis (fan-out; concatenated, not overwritten) ---
    findings: Annotated[list[Finding], operator.add]
    spans: Annotated[list[Span], operator.add]

    # --- output ---
    report: dict[str, Any]


class AnalysisTask(TypedDict):
    """Payload handed to a single `analysis_node` invocation via `Send`."""

    item: ChecklistItem
    retrieved: list[RetrievedChunk]
    transcript: str
    guidance: list[RegulationPassage]
