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


class Chunk(TypedDict, total=False):
    chunk_id: str
    text: str
    # Offsets into the canonical transcript string. Always present.
    start_char: int
    end_char: int
    # Only present when the source had timing (the call_id dataset path).
    start_seconds: int
    end_seconds: int


class RetrievedChunk(Chunk, total=False):
    score: float


class ChecklistItem(TypedDict, total=False):
    id: str
    question: str
    evidence_query: str


class Span(TypedDict, total=False):
    """A verbatim substring of the transcript, located by character offset."""

    text: str
    start: int
    end: int
    label: SpanLabel
    checklist_id: str
    chunk_id: str | None
    timestamp: str | None


class Finding(TypedDict, total=False):
    checklist_id: str
    question: str
    verdict: Verdict
    reason: str
    confidence: Literal["high", "low"]
    spans: list[Span]
    # Kept for the CLI printer and existing consumers: the first verified span.
    evidence_quote: str | None
    timestamp: str | None
    chunk_id: str | None
    retrieved_evidence: list[RetrievedChunk]


class AnalyseInput(TypedDict, total=False):
    """The public API body — what `POST /runs/wait` accepts as `input`.

    Supply `transcript` (a paragraph or a whole transcript) and `checklist`.
    `call_id` is the legacy fallback that loads a call from the synthetic dataset.
    """

    transcript: str
    checklist: list[ChecklistItem]
    call_id: str


class ComplianceState(TypedDict, total=False):
    """State for the top-level graph."""

    # --- set by the caller ---
    call_id: str

    # --- ingestion ---
    transcript: str  # the canonical text every span offset refers to
    call: dict[str, Any]
    chunks: list[Chunk]
    checklist: list[ChecklistItem]

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
