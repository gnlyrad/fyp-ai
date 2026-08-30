"""Transcript ingestion and normalisation stage of the backend pipeline."""
from __future__ import annotations

from typing import Any


def normalise_transcript(utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clean transcript text and preserve timing/speaker labels for evidence use."""
    normalised: list[dict[str, Any]] = []
    for index, utterance in enumerate(utterances):
        text = " ".join(str(utterance.get("text", "")).split())
        if not text:
            continue
        start = int(utterance.get("start_seconds", index))
        end = max(start, int(utterance.get("end_seconds", start)))
        normalised.append({
            "start_seconds": start,
            "end_seconds": end,
            "speaker": str(utterance.get("speaker", "Unknown")).strip() or "Unknown",
            "text": text,
        })
    if not normalised:
        raise ValueError("The transcript has no usable utterances.")
    return normalised
