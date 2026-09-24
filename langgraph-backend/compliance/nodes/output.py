"""`output`: persist the final call review.

Final intended behaviour (CLAUDE.md): write the call, segments, results,
red flags, orders, actions, and escalation to PostgreSQL, storing rule
chunks once and referencing them by id. Behind a `ResultWriter` protocol so
a Postgres writer can be swapped in later without touching the graph.

Step-0: only a `JsonFileWriter` exists, writing `<out_dir>/<call_id>.json`.
Selected via `COMPLIANCE_WRITER` (default "json"; anything else currently
falls back to json, since no other writer is implemented yet).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from compliance.state import ComplianceState


class ResultWriter(Protocol):
    def write(self, state: ComplianceState) -> None: ...


class JsonFileWriter:
    """Writes `<out_dir>/<call_id>.json` with the final aggregated state."""

    def __init__(self, out_dir: str = "output") -> None:
        self.out_dir = Path(out_dir)

    def write(self, state: ComplianceState) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        call_id = state.get("call_id", "unknown")
        payload = {
            "call_id": call_id,
            "segments": state.get("segments", []),
            "quality_flags": state.get("quality_flags", []),
            "call_intent": state.get("call_intent"),
            "orders": _dump(state.get("orders", [])),
            "results": _dump(state.get("final_results", [])),
            "red_flags": _dump(state.get("verified_red_flags") or state.get("red_flags", [])),
            "actions": _dump(state.get("actions", [])),
            "escalation": state.get("escalation"),
            "summary": state.get("summary", {}),
            "errors": state.get("errors", []),
        }
        path = self.out_dir / f"{call_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _dump(items: list) -> list:
    return [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in items]


def _get_writer() -> ResultWriter:
    # Only "json" is implemented in step 0; other values fall back to it.
    kind = os.getenv("COMPLIANCE_WRITER", "json")
    out_dir = os.getenv("COMPLIANCE_OUTPUT_DIR", "output")
    if kind == "json":
        return JsonFileWriter(out_dir=out_dir)
    return JsonFileWriter(out_dir=out_dir)


def output_node(state: ComplianceState) -> dict:
    writer = _get_writer()
    writer.write(state)
    return {}
