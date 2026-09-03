"""Ingestion node: normalise the input into one canonical transcript string plus
retrievable chunks.

Every chunk carries `start_char`/`end_char` offsets into that canonical string,
which is what lets the analysis node turn a model-supplied quote into an exact
substring location in the caller's own text.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from ..prompts.checklist import CHECKLIST
from ..state import Chunk, ComplianceState

DATA_PATH = Path(
    os.getenv(
        "CLEARCALL_DATA_PATH",
        Path(__file__).resolve().parents[3] / "backend" / "data" / "synthetic_conversations.json",
    )
)

WINDOW_SIZE = int(os.getenv("CLEARCALL_CHUNK_WINDOW", "4"))
SENTENCES_PER_CHUNK = int(os.getenv("CLEARCALL_CHUNK_SENTENCES", "3"))

# Sentence boundary: terminal punctuation followed by whitespace.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


def load_call(call_id: str) -> dict[str, Any]:
    calls = json.loads(DATA_PATH.read_text(encoding="utf-8"))["calls"]
    for call in calls:
        if call["call_id"] == call_id:
            return call
    raise KeyError(f"call_id {call_id!r} not found in {DATA_PATH}")


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


def flatten_utterances(
    utterances: list[dict], window_size: int = WINDOW_SIZE
) -> tuple[str, list[Chunk]]:
    """Legacy dataset path: one canonical string plus timed, offset-bearing chunks.

    The rendering (`Speaker: text`, space-joined) matches what the pipeline has
    always fed the model; the only addition is the offsets.
    """
    parts: list[str] = []
    bounds: list[tuple[int, int]] = []
    position = 0
    for utterance in utterances:
        rendered = f'{utterance["speaker"]}: {utterance["text"]}'
        bounds.append((position, position + len(rendered)))
        parts.append(rendered)
        position += len(rendered) + 1  # the joining space

    transcript = " ".join(parts)

    chunks: list[Chunk] = []
    for index in range(0, len(utterances), window_size):
        window = utterances[index:index + window_size]
        start = bounds[index][0]
        end = bounds[index + len(window) - 1][1]
        chunks.append({
            "chunk_id": f"chunk-{index // window_size + 1:02d}",
            "start_char": start,
            "end_char": end,
            "text": transcript[start:end],
            "start_seconds": window[0]["start_seconds"],
            "end_seconds": window[-1]["end_seconds"],
        })
    return transcript, chunks


def ingestion_node(state: ComplianceState) -> dict[str, Any]:
    """Resolve the request into a transcript, its chunks, and the checklist to run.

    A caller-supplied `transcript` wins; `call_id` is the fallback that loads a
    call from the synthetic dataset.
    """
    checklist = list(state.get("checklist") or CHECKLIST)

    transcript = (state.get("transcript") or "").strip()
    if transcript:
        return {
            "transcript": transcript,
            "chunks": chunk_text(transcript),
            "checklist": checklist,
        }

    call = state.get("call")
    if call is None:
        call_id = state.get("call_id")
        if not call_id:
            raise ValueError(
                "Provide either 'transcript' (the text to analyse) or 'call_id' "
                "(a call in the synthetic dataset)."
            )
        call = load_call(call_id)

    transcript, chunks = flatten_utterances(call["utterances"])
    return {
        "transcript": transcript,
        "call": call,
        "chunks": chunks,
        "checklist": checklist,
    }
