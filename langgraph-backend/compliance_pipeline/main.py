"""Local entry point: compile the graph and run one transcript through it.

The same graph is served over HTTP by `langgraph dev`; this is the offline path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .graph import graph


def analyse(
    call_id: str | None = None,
    transcript: str | None = None,
    checklist: list[dict] | None = None,
) -> dict:
    """Run one call or one block of text through the pipeline; return the state."""
    request: dict = {}
    if transcript:
        request["transcript"] = transcript
    if call_id:
        request["call_id"] = call_id
    if checklist:
        request["checklist"] = checklist
    return graph.invoke(request)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the compliance pipeline over one call.")
    parser.add_argument("call_id", nargs="?", default=None,
                        help="A call in the synthetic dataset. Ignored when --transcript-file is given.")
    parser.add_argument("--transcript-file", type=Path,
                        help="A paragraph or full transcript to analyse instead of a dataset call.")
    parser.add_argument("--checklist-file", type=Path,
                        help="JSON list of {id, question} to run instead of the built-in checklist.")
    parser.add_argument("--json", action="store_true", help="Print the full state as JSON.")
    args = parser.parse_args()

    transcript = args.transcript_file.read_text(encoding="utf-8") if args.transcript_file else None
    checklist = json.loads(args.checklist_file.read_text(encoding="utf-8")) if args.checklist_file else None
    call_id = args.call_id or (None if transcript else "CC-3001")

    state = analyse(call_id=call_id, transcript=transcript, checklist=checklist)

    if args.json:
        print(json.dumps(state, indent=2))
        return

    report = state["report"]
    s = report["summary"]
    print(f'{report.get("call_id") or "(text)"} - {report.get("title") or ""}')
    print(f'  pass={s["pass"]} fail={s["fail"]} unclear={s["unclear"]} n/a={s["not_applicable"]}'
          f'  requires_review={s["requires_review"]}')
    for f in report["findings"]:
        mark = {"Pass": "+", "Fail": "!", "Unclear": "?", "N/A": "-"}[f["verdict"]]
        print(f'  {mark} {f["checklist_id"]:16} {f["verdict"]:8} {(f.get("timestamp") or "--:--")}  {f["reason"][:70]}')
    if report["spans"]:
        print(f'  spans: {s["compliant_spans"]} compliant, {s["violation_spans"]} violation')
        for span in report["spans"]:
            print(f'    [{span["label"]:9}] {span["start"]:5}-{span["end"]:<5} {span["text"][:60]!r}')


if __name__ == "__main__":
    main()
