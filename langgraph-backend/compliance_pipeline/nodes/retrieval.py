"""Retrieval node: gather the evidence each checklist item needs.

How much of the transcript an item gets depends on its `question_type`, because
a top-k selection is the wrong answer for some questions:

  targeted      - top-k semantic matches (embedding cosine similarity)
  deterministic - a wider k, plus any chunk literally containing one of the
                  item's terms, so a presence check cannot miss the line
  exhaustive    - the whole transcript, in order; "every occurrence" cannot be
                  answered from three chunks

Scoring is cosine similarity over Bedrock embeddings (the same
`BEDROCK_EMBEDDING_MODEL` used for the regulatory corpus in
`langchain-backend/settings.py`), not lexical overlap — a paraphrase ("no rush"
for "urgency", "assured" for "guaranteed") has to be caught by meaning, not by
shared substrings. If Bedrock is unreachable for a run, retrieval falls back to
a lexical bag-of-words scorer so the pipeline still produces verdicts rather
than erroring out.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from typing import Any, Iterable

from langchain_aws import BedrockEmbeddings

from ..state import (
    DEFAULT_QUESTION_TYPE,
    ChecklistItem,
    Chunk,
    ComplianceState,
    RetrievedChunk,
)

TOKEN_RE = re.compile(r"[a-zA-Z]+")
STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "is", "was",
    "are", "i", "you", "we", "it", "that", "this", "with", "my", "your",
}
TOP_K = int(os.getenv("CLEARCALL_TOP_K", "3"))
# A chunk with too little in common with the query is not evidence. Without this
# floor, `top_k` >= len(chunks) hands every checklist item the whole transcript,
# which turns retrieval into a no-op and lets the model quote anything at all.
MIN_RELEVANCE = float(os.getenv("CLEARCALL_MIN_RELEVANCE", "0.25"))
# A fixed phrase can sit anywhere in the call, so presence checks cast wider than
# the judgement questions that only need the single best passage.
DETERMINISTIC_TOP_K = int(os.getenv("CLEARCALL_DETERMINISTIC_TOP_K", "5"))

EMBEDDING_MODEL = os.getenv("BEDROCK_EMBEDDING_MODEL", "cohere.embed-english-v3")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

_embeddings = None


def embedding_client() -> BedrockEmbeddings:
    """The shared Bedrock embeddings client, built on first use."""
    global _embeddings
    if _embeddings is None:
        _embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL, region_name=AWS_REGION)
    return _embeddings


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


TERM_FORMS = {
    "objectives": "objective", "fees": "fee", "charges": "charge",
    "risks": "risk", "terms": "term", "rights": "right",
    "cancellation": "cancel", "cancellations": "cancel", "cancelled": "cancel",
    "canceling": "cancel", "cancelling": "cancel", "canceled": "cancel",
    "recommendation": "recommend", "recommendations": "recommend", "recommended": "recommend",
    "guaranteed": "guarantee", "guarantees": "guarantee", "guaranteeing": "guarantee",
    "assured": "assure", "assurances": "assure", "written": "writing",
}


def tokens(text: str) -> list[str]:
    # Hyphens/dashes become word boundaries; only search text is normalised.
    return [TERM_FORMS.get(t.lower(), t.lower()) for t in TOKEN_RE.findall(text)
            if t.lower() not in STOP_WORDS]


def retrieve_lexical(
    query: str,
    chunks: Iterable[Chunk],
    top_k: int = TOP_K,
    min_relevance: float = 0.01,
) -> list[RetrievedChunk]:
    """Bag-of-words fallback, used only when the embedding call itself fails.

    An empty result means no lexical match was found, not that the required
    statement is absent. Analysis uses full-call coverage before an omission.
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


def retrieve(
    query: str,
    chunks: list[Chunk],
    chunk_vectors: list[list[float]] | None,
    top_k: int = TOP_K,
    min_relevance: float = MIN_RELEVANCE,
    diagnostics: dict | None = None,
) -> list[RetrievedChunk]:
    """Return the chunks closest in meaning to the query, best first.

    `chunk_vectors` is `None` exactly when embedding this run's chunks already
    failed once (see `retrieval_node`), in which case this falls back to lexical
    matching instead of making — and likely re-failing — another Bedrock call.
    """
    if chunk_vectors is None:
        if diagnostics is not None:
            diagnostics.update(method="lexical_fallback", fallback_reason="chunk_embedding_unavailable")
        return retrieve_lexical(query, chunks, top_k)

    try:
        query_vector = embedding_client().embed_query(query)
    except Exception as exc:
        # The batch chunk embedding above succeeded, but this item's query call
        # didn't (e.g. a transient throttle) — fall back for this item alone
        # rather than failing the whole run over one query.
        if diagnostics is not None:
            diagnostics.update(method="lexical_fallback", fallback_reason=f"query_embedding_failed:{type(exc).__name__}")
        return retrieve_lexical(query, chunks, top_k)

    if diagnostics is not None:
        diagnostics.update(method="semantic", fallback_reason=None)

    results: list[RetrievedChunk] = []
    for chunk, vector in zip(chunks, chunk_vectors):
        score = cosine_similarity(query_vector, vector)
        if score < min_relevance:
            continue
        results.append({**chunk, "score": round(score, 4)})
    return sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]


# Speaker identity is provided by Transcribe; role is caller-supplied metadata.
# Unmapped speaker labels remain neutral in the model context.


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


def chunk_speaker_role(chunk: Chunk) -> str | None:
    """The chunk's resolved role ("RM"/"Customer"), or None when it carries no
    `speaker_label` (a plain-text transcript, not diarized turns)."""
    speaker = chunk.get("speaker_label")
    if not speaker:
        return None
    return chunk.get("speaker_role") or f"{speaker} (role unverified)"


def build_context(retrieved: list[RetrievedChunk]) -> str:
    lines = []
    for r in retrieved:
        prefix = " ".join(filter(None, [
            f'[{r["chunk_id"]}]',
            f'[{chunk_timestamp(r)}]' if chunk_timestamp(r) else None,
            f'{chunk_speaker_role(r)}:' if chunk_speaker_role(r) else None,
        ]))
        lines.append(f'{prefix} {r["text"]}' if prefix else r["text"])
    return "\n\n".join(lines)


def retrieve_targeted(
    item: ChecklistItem, chunks: list[Chunk], chunk_vectors: list[list[float]] | None,
    diagnostics: dict | None = None,
) -> list[RetrievedChunk]:
    """The single best passage or two — enough to judge how well it was done."""
    return retrieve(item.get("evidence_query") or item["question"], chunks, chunk_vectors, diagnostics=diagnostics)


def retrieve_deterministic(
    item: ChecklistItem, chunks: list[Chunk], chunk_vectors: list[list[float]] | None,
    diagnostics: dict | None = None,
) -> list[RetrievedChunk]:
    """Wider top-k, plus every chunk that literally contains one of the terms.

    Add normalised term matches to the semantic candidates. These candidates
    alone cannot establish absence; analysis performs that check over the call.
    """
    query = item.get("evidence_query") or item["question"]
    scored = retrieve(query, chunks, chunk_vectors, top_k=DETERMINISTIC_TOP_K, diagnostics=diagnostics)
    by_id = {chunk["chunk_id"]: chunk for chunk in scored}
    query_terms = set(tokens(query))
    for chunk in chunks:
        if query_terms.intersection(tokens(chunk["text"])):
            by_id.setdefault(chunk["chunk_id"], {**chunk, "score": 0.0})
    return sorted(by_id.values(), key=lambda c: c["start_char"])


def retrieve_exhaustive(
    item: ChecklistItem, chunks: list[Chunk], chunk_vectors: list[list[float]] | None,
    diagnostics: dict | None = None,
) -> list[RetrievedChunk]:
    """The whole transcript, in order. Ranking would defeat the question."""
    if diagnostics is not None:
        diagnostics.update(method="full_transcript", fallback_reason=None)
    return [{**chunk, "score": 1.0} for chunk in chunks]


RETRIEVERS = {
    "targeted": retrieve_targeted,
    "deterministic": retrieve_deterministic,
    "exhaustive": retrieve_exhaustive,
}


def embed_chunks(chunks: list[Chunk]) -> list[list[float]] | None:
    """Embed every transcript chunk once, shared by every checklist item's query.

    `None` means embedding failed for this run (no AWS credentials, Bedrock
    throttled, etc.) — callers fall back to lexical retrieval rather than
    attempting, and likely re-failing, one Bedrock call per checklist item.
    """
    if not chunks:
        return []
    try:
        return embedding_client().embed_documents([chunk["text"] for chunk in chunks])
    except Exception:
        return None


def retrieval_node(state: ComplianceState) -> dict[str, Any]:
    """Retrieve evidence for every checklist item in one pass.

    Chunks are embedded once here rather than inside the fan-out, both because
    only the LLM call is worth parallelising and because every checklist item's
    query is scored against the same chunk vectors — embedding them once and
    reusing the vectors is a large saving over one embedding call per item.
    """
    chunks = state["chunks"]
    active = [item for item in state["checklist"]
              if state.get("applicability", {}).get(item["id"], {}).get("status", "applicable") == "applicable"]
    chunk_vectors = embed_chunks(chunks) if any(i.get("question_type") != "exhaustive" for i in active) else None
    evidence = {}
    diagnostics = {}
    for item in state["checklist"]:
        detail = {"query": item.get("evidence_query") or item["question"],
                  "total_chunks": len(chunks), "method": "skipped", "fallback_reason": None}
        diagnostics[item["id"]] = detail
        if item not in active:
            evidence[item["id"]] = []
            detail.update(selected_chunk_ids=[], coverage="not_reviewed")
            continue
        retriever = RETRIEVERS.get(item.get("question_type") or DEFAULT_QUESTION_TYPE,
                                   retrieve_targeted)
        evidence[item["id"]] = retriever(item, chunks, chunk_vectors, diagnostics=detail)
        detail["selected_chunk_ids"] = [c["chunk_id"] for c in evidence[item["id"]]]
        detail["coverage"] = "full_transcript" if len(evidence[item["id"]]) == len(chunks) else "selected_chunks"
    return {"evidence": evidence, "retrieval_diagnostics": diagnostics}
