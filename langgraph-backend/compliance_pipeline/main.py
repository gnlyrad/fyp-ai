"""CLI entry point: run the graph on one AWS Transcribe output, given as an
S3 URL, a local Transcribe job file, or a request file shaped like
`{"assistant_id": "compliance", "input": {...}}`.

Usage: python -m compliance_pipeline.main <s3-url | file.json> [--out DIR] [--quiet]

Accepts either:
- an S3 URL (`s3://bucket/key.json`, an S3 HTTPS URL, or a presigned URL):
  passed through as `s3_url`, and `ingest` fetches it,
- a full AWS Transcribe job JSON (`{"jobName": ..., "results": {...}}`), or
- a request file with an `"input"` key (`{"input": {"s3_url": "s3://..."}}`
  or `{"input": {"results": {...}, "checklist": [...]}}`, matching what a
  caller posts to the graph).

After the run, prints a human-readable report (`compliance_pipeline.nodes.output.
render_report`) built from the same payload shape the writer persisted
(`compliance_pipeline.nodes.output.build_payload`), so what you see on the CLI
matches what's on disk.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from compliance_pipeline.graph import graph
from compliance_pipeline.nodes.output import build_payload, render_report


def _build_graph_input(payload: dict) -> dict:
    """Normalise a loaded JSON file into the graph's `InputState` shape."""
    if "input" in payload:
        # Request file: {"assistant_id": ..., "input": {"results": ..., "checklist": ...}}
        body = payload["input"] or {}
        graph_input: dict = {}
        if "results" in body:
            graph_input["results"] = body["results"]
        if "checklist" in body:
            graph_input["checklist"] = body["checklist"]
        for key in ("s3_url", "raw_transcript", "call_id", "speaker_roles"):
            if key in body:
                graph_input[key] = body[key]
        return graph_input

    # Otherwise treat the whole file as a Transcribe job JSON (or its
    # "results" object).
    return {"raw_transcript": payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", help="S3 URL of a Transcribe output, or path to a Transcribe job JSON or a request JSON file.")
    parser.add_argument("--out", default="output", help="Output directory for the written JSON.")
    parser.add_argument("--quiet", action="store_true", help="Print only the summary, not the full report.")
    args = parser.parse_args()

    os.environ["COMPLIANCE_OUTPUT_DIR"] = args.out

    if args.transcript.startswith(("s3://", "https://", "http://")):
        graph_input = {"s3_url": args.transcript}
    else:
        payload_in = json.loads(Path(args.transcript).read_text(encoding="utf-8"))
        graph_input = _build_graph_input(payload_in)
    result = graph.invoke(graph_input)

    payload = build_payload(result)
    print(render_report(payload, quiet=args.quiet))

    writer_kind = os.getenv("COMPLIANCE_WRITER", "json")
    if writer_kind == "json":
        out_path = Path(args.out) / f"{payload.get('call_id', 'unknown')}.json"
        print(f"\nOutput: {out_path}")
    else:
        print(f"\nOutput writer: {writer_kind}")


if __name__ == "__main__":
    main()
