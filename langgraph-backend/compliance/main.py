"""CLI entry point: run the graph on one AWS Transcribe job file, or on a
request file shaped like `{"assistant_id": "compliance", "input": {...}}`.

Usage: python -m compliance.main <file.json> [--out DIR] [--client-id ID]

Accepts either:
- a full AWS Transcribe job JSON (`{"jobName": ..., "results": {...}}`), or
- a request file with an `"input"` key (`{"input": {"results": {...},
  "checklist": [...]}}`, matching what a caller posts to the graph).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from compliance.graph import graph


def _build_graph_input(payload: dict, client_id: str | None) -> dict:
    """Normalise a loaded JSON file into the graph's `InputState` shape."""
    extra = {"client_id": client_id} if client_id else {}

    if "input" in payload:
        # Request file: {"assistant_id": ..., "input": {"results": ..., "checklist": ...}}
        body = payload["input"] or {}
        graph_input: dict = {}
        if "results" in body:
            graph_input["results"] = body["results"]
        if "checklist" in body:
            graph_input["checklist"] = body["checklist"]
        for key in ("raw_transcript", "call_id", "client_id", "speaker_roles"):
            if key in body:
                graph_input[key] = body[key]
        return {**graph_input, **extra}

    # Otherwise treat the whole file as a Transcribe job JSON (or its
    # "results" object).
    return {"raw_transcript": payload, **extra}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript_path", help="Path to a Transcribe job JSON or a request JSON file.")
    parser.add_argument("--out", default="output", help="Output directory for the written JSON.")
    parser.add_argument("--client-id", default=None, help="Client id, if known.")
    args = parser.parse_args()

    os.environ["COMPLIANCE_OUTPUT_DIR"] = args.out

    payload = json.loads(Path(args.transcript_path).read_text(encoding="utf-8"))
    graph_input = _build_graph_input(payload, args.client_id)
    result = graph.invoke(graph_input)

    print(f"call_id : {result.get('call_id')}")
    print(f"summary : {json.dumps(result.get('summary', {}), indent=2)}")


if __name__ == "__main__":
    main()
