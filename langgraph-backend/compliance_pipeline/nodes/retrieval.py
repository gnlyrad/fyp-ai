"""Retrieval node: gather the evidence each checklist item needs.

How much of the transcript an item gets depends on its `question_type`, because
a top-k selection is the wrong answer for some questions:

  targeted      - top-k lexical matches (the original behaviour)
  deterministic - a wider k, plus any chunk literally containing one of the
                  item's terms, so a presence check cannot miss the line
  exhaustive    - the whole transcript, in order; "every occurrence" cannot be
                  answered from three chunks
  audio_metrics - nothing, there is no textual evidence to find

The scoring itself is a deliberately lightweight stand-in for an embedding model
plus vector store. Swapping in Azure AI Search or Bedrock Knowledge Bases means
replacing `retrieve` only; the node contract is unchanged.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Iterable

from ..state import (
    DEFAULT_QUESTION_TYPE,
    ChecklistItem,
    Chunk,
    ComplianceState,
    RetrievedChunk,
)
from .tools import chunks_with_phrase

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]+")
STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "is", "was",
    "are", "i", "you", "we", "it", "that", "this", "with", "my", "your",
}
TOP_K = int(os.getenv("CLEARCALL_TOP_K", "3"))
# A chunk sharing no meaningful term with the query is not evidence. Without this
# floor, `top_k` >= len(chunks) hands every checklist item the whole transcript,
# which turns retrieval into a no-op and lets the model quote anything at all.
MIN_RELEVANCE = float(os.getenv("CLEARCALL_MIN_RELEVANCE", "0.01"))
# A fixed phrase can sit anywhere in the call, so presence checks cast wider than
# the judgement questions that only need the single best passage.
DETERMINISTIC_TOP_K = int(os.getenv("CLEARCALL_DETERMINISTIC_TOP_K", "5"))


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOP_WORDS]


def retrieve(
    query: str,
    chunks: Iterable[Chunk],
    top_k: int = TOP_K,
    min_relevance: float = MIN_RELEVANCE,
) -> list[RetrievedChunk]:
    """Return the chunks that actually share terms with the query, best first.

    Returning an empty list is a valid answer: it means the transcript contains
    nothing addressing this checklist item, and the analysis node will report
    `Unclear` rather than invite the model to quote something unrelated.
    """
    q = Counter(tokens(query))
    results: list[RetrievedChunk] = []
    for chunk in chunks:
        doc = Counter(tokens(chunk["text"]))
        overlap = sum(min(q[t], doc[t]) for t in q)
        relevance = overlap / max(1, len(q))
        if relevance < min_relevance:
            continue
        # Lexical overlap with a small preference for concise chunks.
        score = relevance + (0.05 / max(1, len(doc)))
        results.append({**chunk, "score": round(score, 4)})
    return sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]


def format_timestamp(seconds: float) -> str:
    """`mm:ss` — the form an analyst reads back against the recording."""
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def chunk_timestamp(chunk: Chunk) -> str | None:
    """`mm:ss-mm:ss` when the source carried timing, else None (pasted text)."""
    start, end = chunk.get("start_seconds"), chunk.get("end_seconds")
    if start is None or end is None:
        return None
    return f"{format_timestamp(start)}-{format_timestamp(end)}"


def build_context(retrieved: list[RetrievedChunk]) -> str:
    lines = []
    for r in retrieved:
        stamp = chunk_timestamp(r)
        lines.append(f'[{stamp}] {r["text"]}' if stamp else r["text"])
    return "\n\n".join(lines)


def retrieve_targeted(item: ChecklistItem, chunks: list[Chunk]) -> list[RetrievedChunk]:
    """The single best passage or two — enough to judge how well it was done."""
    return retrieve(item.get("evidence_query") or item["question"], chunks)


def retrieve_deterministic(item: ChecklistItem, chunks: list[Chunk]) -> list[RetrievedChunk]:
    """Wider top-k, plus every chunk that literally contains one of the terms.

    The literal pass is the point: a presence check that answers Fail because the
    line ranked sixth is worse than no check at all. Union, kept in transcript
    order so the model reads the call the way it happened.
    """
    query = item.get("evidence_query") or item["question"]
    scored = retrieve(query, chunks, top_k=DETERMINISTIC_TOP_K)
    by_id = {chunk["chunk_id"]: chunk for chunk in scored}
    for chunk in chunks_with_phrase(tokens(query), chunks):
        by_id.setdefault(chunk["chunk_id"], {**chunk, "score": 0.0})
    return sorted(by_id.values(), key=lambda c: c["start_char"])


def retrieve_exhaustive(item: ChecklistItem, chunks: list[Chunk]) -> list[RetrievedChunk]:
    """The whole transcript, in order. Ranking would defeat the question."""
    return [{**chunk, "score": 1.0} for chunk in chunks]


def retrieve_audio_metrics(item: ChecklistItem, chunks: list[Chunk]) -> list[RetrievedChunk]:
    """Nothing: talk-over and pace are not in the words."""
    return []


RETRIEVERS = {
    "targeted": retrieve_targeted,
    "deterministic": retrieve_deterministic,
    "exhaustive": retrieve_exhaustive,
    "audio_metrics": retrieve_audio_metrics,
}


def retrieval_node(state: ComplianceState) -> dict[str, Any]:
    """Retrieve evidence for every checklist item in one pass.

    Batched here rather than inside the fan-out because lexical retrieval is
    cheap and local; only the LLM call is worth parallelising.
    """
    chunks = state["chunks"]
    evidence = {}
    for item in state["checklist"]:
        retriever = RETRIEVERS.get(item.get("question_type") or DEFAULT_QUESTION_TYPE,
                                   retrieve_targeted)
        evidence[item["id"]] = retriever(item, chunks)
    return {"evidence": evidence}
