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
from ..state import AudioSegment, Chunk, ComplianceState

SENTENCES_PER_CHUNK = int(os.getenv("CLEARCALL_CHUNK_SENTENCES", "3"))

# Matches the separator chunks_from_segments joins turns with below.
_TURN_SEPARATOR = "\n\n"

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


def chunks_from_segments(segments: list[AudioSegment]) -> tuple[str, list[Chunk]]:
    """Build the flat transcript and its chunks from diarized Transcribe turns.

    One chunk per turn — no sentence-splitting, since a turn boundary is
    already the meaningful unit here. `text` is the turn's speech only, never
    a reassembled string with a role label baked in: that keeps a located
    quote a clean, displayable excerpt (no "RM: " prefix to strip later) and
    keeps `transcript[start:end] == text` holding exactly the same way it does
    for a plain-text transcript. The speaker's role is still made available to
    the model — `retrieval.build_context` prepends it to the *prompt* context,
    same as it already does for timestamps — just not written into the
    transcript itself. Each chunk also carries the turn's real timestamps
    (`start_seconds`/`end_seconds`) and raw `speaker_label` (e.g. "spk_0"),
    for a caller that wants to attribute or display evidence by speaker.
    """
    parts: list[str] = []
    chunks: list[Chunk] = []
    cursor = 0
    for index, segment in enumerate(segments):
        text = segment.get("transcript", "").strip()
        start = cursor
        end = start + len(text)
        chunks.append({
            "chunk_id": f"chunk-{index + 1:02d}",
            "start_char": start,
            "end_char": end,
            "text": text,
            "start_seconds": float(segment["start_time"]),
            "end_seconds": float(segment["end_time"]),
            "speaker_label": segment.get("speaker_label", ""),
        })
        parts.append(text)
        cursor = end + len(_TURN_SEPARATOR)
    return _TURN_SEPARATOR.join(parts), chunks


def ingestion_node(state: ComplianceState) -> dict[str, Any]:
    """Resolve the request into a transcript, its chunks, and the checklist to run."""
    segments = state.get("audio_segments")
    if segments:
        transcript, chunks = chunks_from_segments(segments)
    else:
        transcript = (state.get("transcript") or "").strip()
        if not transcript:
            raise ValueError(
                "Provide 'transcript' (plain text) or 'audio_segments' "
                "(AWS Transcribe's diarized turns): the call to analyse."
            )
        chunks = chunk_text(transcript)

    return {
        "transcript": transcript,
        "chunks": chunks,
        "checklist": list(state.get("checklist") or CHECKLIST),
    }
