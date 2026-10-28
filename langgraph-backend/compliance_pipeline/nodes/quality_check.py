"""`quality_check`: flag calls that need extra caution downstream.

Pure code, no LLM. Reads `segments` and `dual_channel` from state and
returns `quality_flags`, a list of short string codes (fixed order):

- `short_call`: call duration (last segment `end` minus first segment
  `start`) is known and under `MIN_CALL_SECONDS`.
- `timing_unavailable`: missing/invalid or all-zero timings; informational,
  never interpreted as proof that the conversation was cut short.
- `low_confidence`: word-count-weighted mean of segment `confidence` is
  under `MIN_AVG_CONFIDENCE`. Segments with `confidence` None are skipped;
  if no segment has confidence data, this check is skipped entirely.
- `single_speaker`: fewer than 2 distinct `speaker` values.
- `dual_channel`: informational, set whenever state `dual_channel` is True.
- `empty_transcript`: no segments, or total `word_count` is 0. When this
  fires, the other checks (besides `dual_channel`) are skipped.

Flags never stop the graph; they lower confidence and show on the
dashboard. Thresholds are module-level constants, overridable via the
`COMPLIANCE_MIN_CALL_SECONDS` / `COMPLIANCE_MIN_AVG_CONFIDENCE` env vars,
read once at import time.
"""
from __future__ import annotations

import os
import math

from compliance_pipeline.state import ComplianceState

MIN_CALL_SECONDS = float(os.environ.get("COMPLIANCE_MIN_CALL_SECONDS", 60))
MIN_AVG_CONFIDENCE = float(os.environ.get("COMPLIANCE_MIN_AVG_CONFIDENCE", 0.80))


def quality_check_node(state: ComplianceState) -> dict:
    segments = state.get("segments") or []
    dual_channel = bool(state.get("dual_channel"))

    flags: list[str] = []
    if (state.get("transcription") or {}).get("unclear_turn_ids"):
        flags.append("low_confidence")

    total_words = sum(seg.get("word_count", 0) for seg in segments)
    if not segments or total_words == 0:
        flags.append("empty_transcript")
        if dual_channel:
            flags.append("dual_channel")
        return {"quality_flags": flags}

    try:
        timings = [(float(seg["start"]), float(seg["end"])) for seg in segments]
        duration = timings[-1][1] - timings[0][0]
        usable = (all(math.isfinite(a) and math.isfinite(b) and 0 <= a <= b for a, b in timings)
                  and any(b > a for a, b in timings) and duration > 0)
    except (KeyError, TypeError, ValueError):
        usable, duration = False, 0
    if not usable:
        flags.append("timing_unavailable")
    elif duration < MIN_CALL_SECONDS:
        flags.append("short_call")

    weighted_sum = 0.0
    weighted_words = 0
    for seg in segments:
        confidence = seg.get("confidence")
        if confidence is None:
            continue
        words = seg.get("word_count", 0)
        weighted_sum += confidence * words
        weighted_words += words
    if weighted_words > 0:
        avg_confidence = weighted_sum / weighted_words
        if avg_confidence < MIN_AVG_CONFIDENCE:
            flags.append("low_confidence")

    speakers = {seg.get("speaker") for seg in segments}
    if len(speakers) < 2:
        flags.append("single_speaker")

    if dual_channel:
        flags.append("dual_channel")

    return {"quality_flags": flags}
