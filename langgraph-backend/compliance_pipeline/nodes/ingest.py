"""`ingest`: turn an AWS Transcribe result into speaker+turn transcript segments.

Pure code, no LLM. A "turn" is a maximal run of consecutive speech by the
same speaker; a new turn starts only when the speaker changes. Turns are
numbered 1..N across the whole call and each becomes one `Segment` (stable
id `seg_0001`, ...) -- the evidence id used everywhere downstream.

Source of truth is word-level `results["items"]`, walked in order: speaker
comes from each item's `speaker_label`; punctuation items glue onto the
current turn's last word (no leading space); a turn's `start`/`end` are the
first/last pronunciation `start_time`/`end_time`; `confidence` is the mean
of the turn's pronunciation confidences (3 dp).

Fallbacks, in order, when the primary path is unavailable:
1. Items exist but lack `speaker_label`: look up each pronunciation item's
   speaker by matching its `start_time` into `speaker_labels.segments`;
   punctuation inherits the previous item's speaker.
2. No `items` at all: fall back to `audio_segments`, merging consecutive
   same-speaker segments into one turn and using their `transcript` text
   and start/end (no per-word confidence is available here).
3. Neither `items` nor `audio_segments`: raise `ValueError`.

Dual-channel recordings (`results["channel_labels"]` present) use the
channel label (`ch_0`/`ch_1`) as the speaker and set `dual_channel=True` in
state; note that `speaker_labels.channel_label` alone does *not* mean
dual-channel, only a top-level `channel_labels` does.

Input precedence: `s3_url` (the S3 location of the Transcribe output JSON,
fetched here via `compliance_pipeline.s3`) wins over `results` (the
caller's `input["results"]`), which wins over `raw_transcript` (a full
Transcribe job JSON, or its `results` object). A failed fetch raises
`ValueError`, like any other unusable input.

Also resolves the run's checklist (`config.resolve_checklist`): the
caller's `checklist` if given, else `config/checklist.yaml`. The resolved
items are written to state as `checklist`; an invalid checklist raises
`ValueError` before anything is fetched.
"""
from __future__ import annotations

import hashlib

from compliance_pipeline.config import resolve_checklist
from compliance_pipeline.s3 import fetch_transcribe_json
from compliance_pipeline.schemas import Segment
from compliance_pipeline.state import ComplianceState


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dual_channel_items(results: dict) -> list[dict]:
    """Flatten `channel_labels.channels[*].items` into one chronological list.

    Each returned item is tagged with `speaker_label` set to its channel
    label, so it can be fed through the same speaker-resolution path as a
    normal speaker-diarized item list.
    """
    channel_labels = results.get("channel_labels") or {}
    channels = channel_labels.get("channels") or []
    if not channels:
        # Some jobs may already carry a flat `items` list with a
        # `channel_label` per item; normalise that into `speaker_label`.
        items = []
        for it in results.get("items", []):
            item = dict(it)
            item["speaker_label"] = item.get("channel_label") or item.get("speaker_label")
            items.append(item)
        return items

    tagged: list[tuple[float, dict]] = []
    for ch in channels:
        label = ch.get("channel_label", "unknown")
        last_start = 0.0
        for it in ch.get("items", []):
            item = dict(it)
            item["speaker_label"] = label
            start = _to_float(item.get("start_time"))
            if item.get("type") == "pronunciation" and start is not None:
                last_start = start
                key = start
            else:
                key = last_start
            tagged.append((key, item))
    tagged.sort(key=lambda pair: pair[0])
    return [item for _, item in tagged]


def _resolve_speakers(items: list[dict], results: dict) -> list[str]:
    """One speaker label per item, using `speaker_label` if present, else
    falling back to `speaker_labels.segments` matched by start_time."""
    has_speaker_label = any(
        it.get("speaker_label") for it in items if it.get("type") == "pronunciation"
    )
    if has_speaker_label:
        speakers: list[str] = []
        prev = "unknown"
        for it in items:
            sp = it.get("speaker_label") or prev
            speakers.append(sp)
            if it.get("type") == "pronunciation" and it.get("speaker_label"):
                prev = it["speaker_label"]
        return speakers

    segments = ((results.get("speaker_labels") or {}).get("segments")) or []

    def _find(start_time: float) -> str:
        for seg in segments:
            seg_start = _to_float(seg.get("start_time"))
            seg_end = _to_float(seg.get("end_time"))
            if seg_start is not None and seg_end is not None and seg_start <= start_time <= seg_end:
                return seg.get("speaker_label", "unknown")
        return "unknown"

    speakers = []
    prev = "unknown"
    for it in items:
        start = _to_float(it.get("start_time"))
        if it.get("type") == "pronunciation" and start is not None:
            prev = _find(start)
        speakers.append(prev)
    return speakers


def _group_into_turns(items: list[dict], speakers: list[str]) -> list[dict]:
    """Group items (with resolved speakers) into consecutive same-speaker turns."""
    turns: list[dict] = []
    current = None
    for item, speaker in zip(items, speakers):
        if current is None or speaker != current["speaker"]:
            current = {"speaker": speaker, "words": [], "confidences": [], "start": None, "end": None}
            turns.append(current)

        alt = (item.get("alternatives") or [{}])[0]
        content = alt.get("content", "")
        confidence = _to_float(alt.get("confidence"))

        if item.get("type") == "punctuation":
            if current["words"]:
                current["words"][-1] = current["words"][-1] + content
            else:
                current["words"].append(content)
            continue

        current["words"].append(content)
        start = _to_float(item.get("start_time"))
        end = _to_float(item.get("end_time"))
        if start is not None and current["start"] is None:
            current["start"] = start
        if end is not None:
            current["end"] = end
        if confidence is not None:
            current["confidences"].append(confidence)

    return [t for t in turns if t["start"] is not None]


def _segments_from_items(items: list[dict], results: dict) -> list[Segment]:
    speakers = _resolve_speakers(items, results)
    turns = _group_into_turns(items, speakers)

    segments = []
    for i, turn in enumerate(turns, start=1):
        confidences = turn["confidences"]
        segments.append(Segment(
            id=f"seg_{i:04d}",
            speaker=turn["speaker"],
            turn=i,
            start=turn["start"],
            end=turn["end"],
            text=" ".join(turn["words"]),
            confidence=round(sum(confidences) / len(confidences), 3) if confidences else None,
            word_count=len(turn["words"]),
        ))
    return segments


def _segments_from_audio_segments(audio_segments: list[dict]) -> list[Segment]:
    """Fallback when there are no `items`: merge consecutive same-speaker
    `audio_segments` into turns, using their own transcript/start/end."""
    turns: list[dict] = []
    for seg in audio_segments:
        speaker = seg.get("speaker_label", "unknown")
        text = seg.get("transcript", "")
        start = _to_float(seg.get("start_time")) or 0.0
        end = _to_float(seg.get("end_time")) or start
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] = f"{turns[-1]['text']} {text}".strip()
            turns[-1]["end"] = end
        else:
            turns.append({"speaker": speaker, "text": text, "start": start, "end": end})

    segments = []
    for i, turn in enumerate(turns, start=1):
        segments.append(Segment(
            id=f"seg_{i:04d}",
            speaker=turn["speaker"],
            turn=i,
            start=turn["start"],
            end=turn["end"],
            text=turn["text"],
            confidence=None,
            word_count=len(turn["text"].split()),
        ))
    return segments


def _load_raw(state: ComplianceState) -> tuple[dict, dict]:
    """`(raw, results)`: the full Transcribe job JSON (or `{}` when only a
    bare `results` object is known) and its `results` object.

    Precedence: `s3_url` (fetched here) > `results` > `raw_transcript`.
    """
    s3_url = state.get("s3_url")
    if s3_url:
        raw = fetch_transcribe_json(s3_url)
    elif state.get("results") is not None:
        return {}, state["results"]
    else:
        raw = state.get("raw_transcript") or {}
    return raw, (raw["results"] if "results" in raw else raw)


def ingest_node(state: ComplianceState) -> dict:
    # Resolve the checklist first so a bad one fails before any S3 fetch.
    supplied_checklist = state.get("checklist")
    using_default_checklist = not supplied_checklist
    checklist = resolve_checklist(supplied_checklist)
    raw, raw_results = _load_raw(state)

    dual_channel = bool(raw_results.get("channel_labels"))

    if dual_channel:
        items = _dual_channel_items(raw_results)
    else:
        items = raw_results.get("items") or []

    if items:
        segments = _segments_from_items(items, raw_results)
    else:
        audio_segments = raw_results.get("audio_segments") or []
        if not audio_segments:
            raise ValueError(
                "ingest: no usable transcript data -- results has neither "
                "'items' nor 'audio_segments'"
            )
        segments = _segments_from_audio_segments(audio_segments)

    from compliance_pipeline.nodes._common import format_transcript
    transcript = format_transcript([s.model_dump() for s in segments], with_timestamps=True)

    call_id = state.get("call_id") or raw.get("jobName") or hashlib.sha1(transcript.encode("utf-8")).hexdigest()[:12]

    return {
        "call_id": call_id,
        "segments": [s.model_dump() for s in segments],
        "transcript": transcript,
        "checklist": [item.model_dump() for item in checklist],
        "using_default_checklist": using_default_checklist,
        "dual_channel": dual_channel,
    }
