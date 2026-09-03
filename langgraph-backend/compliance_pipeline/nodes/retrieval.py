"""Retrieval node: lexical scoring of transcript chunks per checklist item.

This is a deliberately lightweight stand-in for an embedding model plus vector
store, mirroring `backend/app/rag.py`. Swapping in Azure AI Search or Bedrock
Knowledge Bases means replacing `retrieve` only; the node contract is unchanged.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Iterable

from ..state import Chunk, ComplianceState, RetrievedChunk

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]+")
STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "is", "was",
    "are", "i", "you", "we", "it", "that", "this", "with", "my", "your",
}
TOP_K = int(os.getenv("CLEARCALL_TOP_K", "3"))


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOP_WORDS]


def format_timestamp(seconds: int) -> str:
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def retrieve(query: str, chunks: Iterable[Chunk], top_k: int = TOP_K) -> list[RetrievedChunk]:
    q = Counter(tokens(query))
    results: list[RetrievedChunk] = []
    for chunk in chunks:
        doc = Counter(tokens(chunk["text"]))
        overlap = sum(min(q[t], doc[t]) for t in q)
        # Lexical overlap with a small preference for concise chunks.
        score = overlap / max(1, len(q)) + (0.05 / max(1, len(doc)))
        results.append({**chunk, "score": round(score, 4)})
    return sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]


def chunk_timestamp(chunk: Chunk) -> str | None:
    """`mm:ss-mm:ss` when the source carried timing, else None (pasted text)."""
    if chunk.get("start_seconds") is None:
        return None
    return f'{format_timestamp(chunk["start_seconds"])}-{format_timestamp(chunk["end_seconds"])}'


def build_context(retrieved: list[RetrievedChunk]) -> str:
    lines = []
    for r in retrieved:
        stamp = chunk_timestamp(r)
        lines.append(f'[{stamp}] {r["text"]}' if stamp else r["text"])
    return "\n\n".join(lines)


def retrieval_node(state: ComplianceState) -> dict[str, Any]:
    """Retrieve evidence for every checklist item in one pass.

    Batched here rather than inside the fan-out because lexical retrieval is
    cheap and local; only the LLM call is worth parallelising.
    """
    chunks = state["chunks"]
    evidence = {
        item["id"]: retrieve(item.get("evidence_query") or item["question"], chunks)
        for item in state["checklist"]
    }
    return {"evidence": evidence}
