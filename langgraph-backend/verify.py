"""Check a pipeline response: verdicts, and that every span is an exact slice.

Usage:
    python verify.py                 # read response.json (from the curl test)
    python verify.py --live          # run the graph in-process instead
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUEST = Path(__file__).with_name("request.json")
RESPONSE = Path(__file__).with_name("response.json")


def load_state(live: bool) -> dict:
    if not live:
        return json.loads(RESPONSE.read_text(encoding="utf-8"))
    from compliance_pipeline.graph import graph
    return graph.invoke(json.loads(REQUEST.read_text(encoding="utf-8"))["input"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Invoke the graph directly.")
    args = parser.parse_args()

    state = load_state(args.live)
    transcript = state["transcript"]

    print("keys:", sorted(state))
    print("checklist:", [i["id"] for i in state["checklist"]])
    print()
    for f in state["findings"]:
        print(f'  {f["checklist_id"]:18} {f["verdict"]:8} {f["confidence"]:5} '
              f'{len(f["spans"])} span(s)  {f["reason"][:58]}')

    print("\nspans:")
    mismatches = []
    for s in state["report"]["spans"]:
        exact = transcript[s["start"]:s["end"]] == s["text"]
        if not exact:
            mismatches.append(s)
        print(f'  [{s["label"]:9}] {s["checklist_id"]:16} {s["start"]:4}-{s["end"]:<4} '
              f'exact={exact}  {s["text"][:60]!r}')

    print("\nOFFSET MISMATCHES:", len(mismatches))
    print("summary:", json.dumps(state["report"]["summary"]))


if __name__ == "__main__":
    main()
