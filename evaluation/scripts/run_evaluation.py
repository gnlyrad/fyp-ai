"""Run the real LangGraph NLP pipeline against the published evaluation dataset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION_ROOT))

from fyp_evaluation_engine.runner import run_evaluation  # noqa: E402
from fyp_evaluation_loader import DatasetError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=EVALUATION_ROOT / "datasets" / "pilot")
    parser.add_argument("--output-root", type=Path, default=EVALUATION_ROOT / "runs")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Evaluate every published call")
    selection.add_argument("--call", dest="call_ids", action="append", help="Call ID; repeat to select several")
    parser.add_argument("--run-id", help="Optional immutable directory name")
    parser.add_argument("--context-mode", choices=["full_transcript"],
                        help="Full transcript (the staged pipeline context mode)")
    parser.add_argument("--max-attempts", type=int, choices=range(1, 6),
                        help="Total attempts per LLM stage; 1 disables application retries")
    args = parser.parse_args(argv)
    try:
        result = run_evaluation(
            args.dataset,
            args.output_root,
            call_ids=None if args.all else args.call_ids,
            run_id=args.run_id,
            progress=print,
            context_mode=args.context_mode,
            analysis_max_attempts=args.max_attempts,
        )
    except (DatasetError, FileExistsError, ValueError) as error:
        print(f"Evaluation setup error: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Evaluation failed before artifacts were persisted: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(f"Run artifacts: {result['run_dir']}")
    print(json.dumps(result["summary"]["overall"], ensure_ascii=False, indent=2))
    return 0 if result["metadata"]["execution_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
