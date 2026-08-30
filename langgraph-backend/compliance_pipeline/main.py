"""Local entry point: compile the graph and run one call through it."""
from __future__ import annotations

import argparse
import json

from .graph import graph


def analyse(call_id: str) -> dict:
    result = graph.invoke({"call_id": call_id})
    return result["report"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the compliance pipeline over one call.")
    parser.add_argument("call_id", nargs="?", default="CC-3001")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    args = parser.parse_args()

    report = analyse(args.call_id)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    s = report["summary"]
    print(f'{report["call_id"]} - {report.get("title")}')
    print(f'  pass={s["pass"]} fail={s["fail"]} unclear={s["unclear"]} n/a={s["not_applicable"]}'
          f'  requires_review={s["requires_review"]}')
    for f in report["findings"]:
        mark = {"Pass": "+", "Fail": "!", "Unclear": "?", "N/A": "-"}[f["verdict"]]
        print(f'  {mark} {f["checklist_id"]:16} {f["verdict"]:8} {(f.get("timestamp") or "--:--")}  {f["reason"][:70]}')


if __name__ == "__main__":
    main()
