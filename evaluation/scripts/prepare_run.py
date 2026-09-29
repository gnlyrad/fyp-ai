"""Validate one prepared call and print the direct LangGraph input; no model call."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fyp_evaluation_loader import DatasetError, graph_input_for_call, load_dataset  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "pilot")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--call", dest="call_id")
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="Required for this scaffold; no graph invocation exists yet")
    args = parser.parse_args(argv)
    if not args.check_only:
        parser.error("This scaffold is read-only. Pass --check-only.")
    try:
        dataset = load_dataset(args.dataset)
        call_ids = dataset["manifest"]["call_ids"] if args.all else [args.call_id]
        for call_id in call_ids:
            graph_input = graph_input_for_call(dataset, call_id)
            print(f"{call_id}: valid direct graph input ({len(graph_input['checklist'])} checklist items)")
            if not args.all:
                print(json.dumps(graph_input, ensure_ascii=False, indent=2))
    except DatasetError as error:
        print(f"Dataset error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
