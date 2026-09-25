"""Rescore a saved run into a NEW run directory without invoking LangGraph or Bedrock."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION_ROOT))

from fyp_evaluation_engine.rescore import rescore_run  # noqa: E402
from fyp_evaluation_loader import DatasetError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True, help="Run ID under --output-root, or a run directory path")
    parser.add_argument("--dataset", type=Path, default=EVALUATION_ROOT / "datasets" / "pilot")
    parser.add_argument("--output-root", type=Path, default=EVALUATION_ROOT / "runs")
    parser.add_argument("--run-id", help="Optional immutable directory name for the derived run")
    args = parser.parse_args(argv)
    source = Path(args.source_run)
    if not source.is_dir():
        source = args.output_root / args.source_run
    try:
        result = rescore_run(source, args.output_root, args.dataset, run_id=args.run_id)
    except (DatasetError, FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"Rescore setup error: {error}", file=sys.stderr)
        return 2
    print(f"Derived run: {result['run_dir']}")
    print(json.dumps(result["summary"]["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
