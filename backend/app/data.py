from __future__ import annotations

import json
import os
from pathlib import Path

DATA_PATH = Path(os.getenv("CLEARCALL_DATA_PATH", Path(__file__).resolve().parents[1] / "data" / "synthetic_conversations.json"))


def load_calls() -> list[dict]:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))["calls"]


def get_call(call_id: str) -> dict:
    for call in load_calls():
        if call["call_id"] == call_id:
            return call
    raise KeyError(call_id)
