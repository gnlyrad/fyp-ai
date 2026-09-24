"""Small helpers shared by more than one node."""
from __future__ import annotations


def format_transcript(segments: list[dict], with_timestamps: bool = False) -> str:
    """Render segments as `[seg_0001] Advisor: text` lines for LLM prompts.

    Falls back to the raw speaker label (e.g. `spk_0`) when `role` has not
    been assigned yet. With `with_timestamps=True`, renders
    `[seg_0001] spk_0 (0.48-16.48): text` instead (used for the human/LLM
    facing `transcript` string built by `ingest`).
    """
    lines = []
    for seg in segments:
        speaker = seg.get("role") or seg.get("speaker") or "unknown"
        if with_timestamps:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            lines.append(f"[{seg['id']}] {speaker} ({start:.2f}-{end:.2f}): {seg.get('text', '')}")
        else:
            lines.append(f"[{seg['id']}] {speaker}: {seg.get('text', '')}")
    return "\n".join(lines)
