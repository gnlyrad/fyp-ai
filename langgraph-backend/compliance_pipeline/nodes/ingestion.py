"""Ingestion node: normalise the submitted transcript into retrievable chunks.

Every chunk carries `start_char`/`end_char` offsets into the transcript, which is
what lets the analysis node turn a model-supplied quote into an exact substring
location in the caller's own text.
"""
from __future__ import annotations

import os
import math
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
            "chunk_id": f"chunk-{index // sentences_per_chunk:02d}",
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
        if not isinstance(segment, dict) or not isinstance(segment.get("transcript"), str):
            raise ValueError(f"audio_segments[{index}].transcript must be text.")
        text = segment["transcript"].strip()
        try:
            begin, finish = float(segment["start_time"]), float(segment["end_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"audio_segments[{index}] needs numeric start_time/end_time.") from exc
        if not text or not math.isfinite(begin) or not math.isfinite(finish) or begin < 0 or finish < begin:
            raise ValueError(f"audio_segments[{index}] has empty text or invalid timing.")
        start = cursor
        end = start + len(text)
        chunks.append({
            "chunk_id": f"chunk-{index:02d}",
            "source_segment_id": segment.get("id", index),
            "start_char": start,
            "end_char": end,
            "text": text,
            "start_seconds": begin,
            "end_seconds": finish,
            "speaker_label": segment.get("speaker_label", ""),
        })
        parts.append(text)
        cursor = end + len(_TURN_SEPARATOR)
    return _TURN_SEPARATOR.join(parts), chunks


def align_word_timings(chunk: Chunk, segment: dict, items: list[dict]) -> list[dict]:
    """Align source items exactly in order; never guess word positions or timings."""
    by_id = {item.get("id", i): item for i, item in enumerate(items)}
    cursor = 0
    aligned = []
    for item_id in segment.get("items", []):
        item = by_id.get(item_id)
        if not item or not item.get("alternatives"):
            return []
        word = item["alternatives"][0].get("content", "")
        if not word:
            return []
        offset = chunk["text"].find(word, cursor)
        if offset < 0 or chunk["text"][cursor:offset].strip():
            return []
        cursor = offset + len(word)
        if item.get("type") != "pronunciation":
            continue
        try:
            begin, finish = float(item["start_time"]), float(item["end_time"])
        except (KeyError, TypeError, ValueError):
            return []
        if (not math.isfinite(begin) or not math.isfinite(finish)
                or begin < chunk["start_seconds"] or finish > chunk["end_seconds"]
                or finish < begin):
            return []
        aligned.append({"start": chunk["start_char"] + offset,
                        "end": chunk["start_char"] + cursor,
                        "start_seconds": begin, "end_seconds": finish})
    if chunk["text"][cursor:].strip():
        return []
    return aligned


def ingestion_node(state: ComplianceState) -> dict[str, Any]:
    """Resolve the request into a transcript, its chunks, and the checklist to run."""
    update: dict[str, Any] = {}
    segments = state.get("audio_segments")
    transcript = state.get("transcript") or ""
    if "results" in state:
        results = state["results"]
        if not isinstance(results, dict):
            raise ValueError("'results' must be the AWS Transcribe results object.")
        # Keep the original object: word timings, speaker_labels, and other
        # Transcribe metadata remain available to callers in the final state.
        update["results"] = results
        segments = results.get("audio_segments")
        if segments is not None and not isinstance(segments, list):
            raise ValueError("'results.audio_segments' must be a list.")
        transcript = ""
        if not segments:
            transcripts = results.get("transcripts")
            if not isinstance(transcripts, list) or not transcripts:
                raise ValueError(
                    "'results' must contain non-empty 'audio_segments' or 'transcripts'."
                )
            first = transcripts[0]
            if not isinstance(first, dict) or not isinstance(first.get("transcript"), str):
                raise ValueError("'results.transcripts[0].transcript' must be text.")
            transcript = first["transcript"]
        update["audio_segments"] = segments or []

    if segments:
        transcript, chunks = chunks_from_segments(segments)
    else:
        transcript = transcript.strip()
        if not transcript:
            raise ValueError(
                "Provide non-empty 'results' (AWS Transcribe), 'audio_segments', "
                "or 'transcript' (plain text): the call to analyse."
            )
        chunks = chunk_text(transcript)

    roles = state.get("speaker_roles") or {}
    if not isinstance(roles, dict) or any(role not in {"RM", "Customer"} for role in roles.values()):
        raise ValueError("speaker_roles must map source speaker labels to 'RM' or 'Customer'.")
    known_speakers = {c.get("speaker_label") for c in chunks if c.get("speaker_label")}
    if set(roles) - known_speakers:
        raise ValueError("speaker_roles contains a label not present in the supplied segments.")
    warnings = []
    for chunk in chunks:
        speaker = chunk.get("speaker_label")
        if speaker in roles:
            chunk["speaker_role"] = roles[speaker]
            chunk["speaker_role_source"] = "caller"
    if known_speakers - set(roles):
        warnings.append("Speaker identities are preserved; some RM/Customer roles are unverified.")
    if segments and state.get("results", {}).get("items"):
        source_items = state["results"]["items"]
        items_by_id = {item.get("id", i): item for i, item in enumerate(source_items)}
        for chunk, segment in zip(chunks, segments):
            speakers = {items_by_id[i].get("speaker_label") for i in segment.get("items", [])
                        if i in items_by_id and items_by_id[i].get("speaker_label")}
            if speakers - {chunk.get("speaker_label")}:
                warnings.append(f"{chunk['chunk_id']}: word speaker labels disagree with the segment; attribution needs review.")
                chunk.pop("speaker_role", None)
                chunk.pop("speaker_role_source", None)
            aligned = align_word_timings(chunk, segment, source_items)
            if aligned:
                chunk["word_timings"] = aligned
            else:
                warnings.append(f"{chunk['chunk_id']}: word timings could not be aligned; using segment timing.")
    return {
        **update,
        "transcript": transcript,
        "chunks": chunks,
        "checklist": list(state.get("checklist") or CHECKLIST),
        "speaker_roles": dict(roles),
        "input_warnings": warnings,
    }
