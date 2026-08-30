"""Ingestion node: load a call and normalise it into retrievable chunks."""
from __future__ import annotations

import json
import os
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


def load_call(call_id: str) -> dict[str, Any]:
    calls = json.loads(DATA_PATH.read_text(encoding="utf-8"))["calls"]
    for call in calls:
        if call["call_id"] == call_id:
            return call
    raise KeyError(f"call_id {call_id!r} not found in {DATA_PATH}")


def build_chunks(utterances: list[dict], window_size: int = WINDOW_SIZE) -> list[Chunk]:
    """Fixed-size sliding windows over utterances, preserving speaker and timing."""
    chunks: list[Chunk] = []
    for index in range(0, len(utterances), window_size):
        window = utterances[index:index + window_size]
        chunks.append({
            "chunk_id": f"chunk-{index // window_size + 1:02d}",
            "start_seconds": window[0]["start_seconds"],
            "end_seconds": window[-1]["end_seconds"],
            "text": " ".join(f'{u["speaker"]}: {u["text"]}' for u in window),
        })
    return chunks


def ingestion_node(state: ComplianceState) -> dict[str, Any]:
    """Resolve `call_id` into the call, its chunks, and the checklist to run."""
    call = state.get("call") or load_call(state["call_id"])
    return {
        "call": call,
        "chunks": build_chunks(call["utterances"]),
        "checklist": list(CHECKLIST),
    }
