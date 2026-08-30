"""Graph state for the compliance pipeline.

`findings` uses `operator.add` so that the fanned-out analysis nodes can each
return a one-element list and have LangGraph concatenate them, rather than the
last writer overwriting the rest.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

Verdict = Literal["Pass", "Fail", "Unclear", "N/A"]


class Chunk(TypedDict):
    chunk_id: str
    start_seconds: int
    end_seconds: int
    text: str


class RetrievedChunk(Chunk):
    score: float


class ChecklistItem(TypedDict, total=False):
    id: str
    question: str
    evidence_query: str


class Finding(TypedDict, total=False):
    checklist_id: str
    question: str
    verdict: Verdict
    reason: str
    confidence: Literal["high", "low"]
    evidence_quote: str | None
    timestamp: str | None
    chunk_id: str | None
    retrieved_evidence: list[RetrievedChunk]


class ComplianceState(TypedDict, total=False):
    """State for the top-level graph."""

    # --- set by the caller ---
    call_id: str

    # --- ingestion ---
    call: dict[str, Any]
    chunks: list[Chunk]
    checklist: list[ChecklistItem]

    # --- retrieval ---
    evidence: dict[str, list[RetrievedChunk]]  # checklist_id -> retrieved chunks

    # --- analysis (fan-out; concatenated, not overwritten) ---
    findings: Annotated[list[Finding], operator.add]

    # --- output ---
    report: dict[str, Any]


class AnalysisTask(TypedDict):
    """Payload handed to a single `analysis_node` invocation via `Send`."""

    item: ChecklistItem
    retrieved: list[RetrievedChunk]
