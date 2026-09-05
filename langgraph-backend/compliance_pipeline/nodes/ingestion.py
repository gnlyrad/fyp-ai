"""Ingestion node: normalise the submitted transcript into retrievable chunks.

Every chunk carries `start_char`/`end_char` offsets into the transcript, which is
what lets the analysis node turn a model-supplied quote into an exact substring
location in the caller's own text.
"""
from __future__ import annotations

import os
import re
from typing import Any

from ..prompts.checklist import CHECKLIST
from ..state import Chunk, ComplianceState

SENTENCES_PER_CHUNK = int(os.getenv("CLEARCALL_CHUNK_SENTENCES", "3"))

# Sentence boundary: terminal punctuation followed by whitespace.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of each sentence, preserving the original offsets."""
    spans: list[tuple[int, int]] = []
    position = 0
    for match in _SENTENCE_BREAK.finditer(text):
        if match.start() > position:
            spans.append((position, match.start()))
        position = match.end()
    if position < len(text):
        spans.append((position, len(text)))
    return spans


def chunk_text(text: str, sentences_per_chunk: int = SENTENCES_PER_CHUNK) -> list[Chunk]:
    """Group sentences into windows, slicing offsets out of the original string.

    The chunk text is always a literal slice of `text`, never reassembled, so a
    quote found inside a chunk maps back to the transcript by simple addition.
    """
    spans = sentence_spans(text)
    if not spans:
        return []

    chunks: list[Chunk] = []
    for index in range(0, len(spans), sentences_per_chunk):
        window = spans[index:index + sentences_per_chunk]
        start, end = window[0][0], window[-1][1]
        chunks.append({
            "chunk_id": f"chunk-{index // sentences_per_chunk + 1:02d}",
            "start_char": start,
            "end_char": end,
            "text": text[start:end],
        })
    return chunks


def ingestion_node(state: ComplianceState) -> dict[str, Any]:
    """Resolve the request into a transcript, its chunks, and the checklist to run."""
    transcript = (state.get("transcript") or "").strip()
    if not transcript:
        raise ValueError("Provide 'transcript': the call text to analyse.")

    return {
        "transcript": transcript,
        "chunks": chunk_text(transcript),
        "checklist": list(state.get("checklist") or CHECKLIST),
    }
