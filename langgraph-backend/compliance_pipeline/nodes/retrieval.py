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
from .tools import chunks_with_phrase

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]+")
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


def tokens(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOP_WORDS]


def retrieve_lexical(
    query: str,
    chunks: Iterable[Chunk],
    top_k: int = TOP_K,
    min_relevance: float = 0.01,
) -> list[RetrievedChunk]:
    """Bag-of-words fallback, used only when the embedding call itself fails.

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


def retrieve(
    query: str,
    chunks: list[Chunk],
    chunk_vectors: list[list[float]] | None,
    top_k: int = TOP_K,
    min_relevance: float = MIN_RELEVANCE,
) -> list[RetrievedChunk]:
    """Return the chunks closest in meaning to the query, best first.

    `chunk_vectors` is `None` exactly when embedding this run's chunks already
    failed once (see `retrieval_node`), in which case this falls back to lexical
    matching instead of making — and likely re-failing — another Bedrock call.
    """
    if chunk_vectors is None:
        return retrieve_lexical(query, chunks, top_k)

    try:
        query_vector = embedding_client().embed_query(query)
    except Exception:
        # The batch chunk embedding above succeeded, but this item's query call
        # didn't (e.g. a transient throttle) — fall back for this item alone
        # rather than failing the whole run over one query.
        return retrieve_lexical(query, chunks, top_k)

    results: list[RetrievedChunk] = []
    for chunk, vector in zip(chunks, chunk_vectors):
        score = cosine_similarity(query_vector, vector)
        if score < min_relevance:
            continue
        results.append({**chunk, "score": round(score, 4)})
    return sorted(results, key=lambda r: r["score"], reverse=True)[:top_k]


# AWS Transcribe's diarization tells us two speakers exist and separates their
# turns; it never tells us which one is staff. This fixed spk_0 -> RM mapping
# is a starting assumption, not something Transcribe guarantees: it holds for
# an outbound call where the RM places the call and speaks first, and silently
# mislabels an inbound call, a call where the customer speaks first, or one
# where the recording channels are swapped. Revisit if either becomes common.
#
# Used only to label the *prompt* context the model reads (so it can attribute
# statements to the right party) — never written into `chunk["text"]` itself,
# so a located quote stays a clean excerpt with no role prefix to strip.
SPEAKER_ROLE = {"spk_0": "RM", "spk_1": "Customer"}
DEFAULT_SPEAKER_ROLE = "Speaker"


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
    return SPEAKER_ROLE.get(speaker, DEFAULT_SPEAKER_ROLE)


def build_context(retrieved: list[RetrievedChunk]) -> str:
    lines = []
    for r in retrieved:
        prefix = " ".join(filter(None, [
            f'[{chunk_timestamp(r)}]' if chunk_timestamp(r) else None,
            f'{chunk_speaker_role(r)}:' if chunk_speaker_role(r) else None,
        ]))
        lines.append(f'{prefix} {r["text"]}' if prefix else r["text"])
    return "\n\n".join(lines)


def retrieve_targeted(
    item: ChecklistItem, chunks: list[Chunk], chunk_vectors: list[list[float]] | None,
) -> list[RetrievedChunk]:
    """The single best passage or two — enough to judge how well it was done."""
    return retrieve(item.get("evidence_query") or item["question"], chunks, chunk_vectors)


def retrieve_deterministic(
    item: ChecklistItem, chunks: list[Chunk], chunk_vectors: list[list[float]] | None,
) -> list[RetrievedChunk]:
    """Wider top-k, plus every chunk that literally contains one of the terms.

    The literal pass is the point: a presence check that answers Fail because the
    line ranked sixth is worse than no check at all. Union, kept in transcript
    order so the model reads the call the way it happened.
    """
    query = item.get("evidence_query") or item["question"]
    scored = retrieve(query, chunks, chunk_vectors, top_k=DETERMINISTIC_TOP_K)
    by_id = {chunk["chunk_id"]: chunk for chunk in scored}
    for chunk in chunks_with_phrase(tokens(query), chunks):
        by_id.setdefault(chunk["chunk_id"], {**chunk, "score": 0.0})
    return sorted(by_id.values(), key=lambda c: c["start_char"])


def retrieve_exhaustive(
    item: ChecklistItem, chunks: list[Chunk], chunk_vectors: list[list[float]] | None,
) -> list[RetrievedChunk]:
    """The whole transcript, in order. Ranking would defeat the question."""
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
    chunk_vectors = embed_chunks(chunks)
    evidence = {}
    for item in state["checklist"]:
        retriever = RETRIEVERS.get(item.get("question_type") or DEFAULT_QUESTION_TYPE,
                                   retrieve_targeted)
        evidence[item["id"]] = retriever(item, chunks, chunk_vectors)
    return {"evidence": evidence}
