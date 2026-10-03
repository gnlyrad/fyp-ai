"""Publish one complete immutable evaluator run to Supabase.

    python -B evaluation/scripts/upload_run.py              # pick a run from a list
    python -B evaluation/scripts/upload_run.py --run-id ID  # publish a specific run
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EVALUATION_ROOT.parent
sys.path.insert(0, str(EVALUATION_ROOT))

from dashboard.local_repository import LocalEvaluationRepository  # noqa: E402
from fyp_evaluation_engine.publication import PublicationError, SupabasePublisher, validate_run_dir  # noqa: E402

SGT = timezone(timedelta(hours=8), "SGT")  # Singapore has no daylight saving


def _sgt(value: str | None) -> str:
    if not value:
        return "unknown time"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.astimezone(SGT).strftime("%d %b %Y %I:%M %p SGT")


def _local_runs() -> list[dict]:
    repository = LocalEvaluationRepository(EVALUATION_ROOT)
    runs = []
    for dataset in repository.list_datasets():
        for run in repository.list_runs(dataset.dataset_id):
            runs.append({"run": run, "dataset": dataset.name})
    runs.sort(key=lambda item: item["run"].evaluated_at or "", reverse=True)
    return runs


def choose_run(publisher: SupabasePublisher) -> str | None:
    """Show local runs and ask which one to publish. Returns a run ID, or None to quit."""
    runs = _local_runs()
    if not runs:
        print("No local runs found in evaluation/runs.")
        return None
    try:
        published = publisher.published_run_ids()
    except Exception:
        published = None  # still usable: the publisher itself detects duplicates
    print("\nLocal runs (newest first):\n")
    for number, item in enumerate(runs, start=1):
        run = item["run"]
        if not run.is_complete:
            note = "partial run, cannot be published"
        elif published is not None and run.run_id in published:
            note = "already published"
        else:
            note = "ready"
        print(f"  [{number}] {_sgt(run.evaluated_at)}  {item['dataset']}  {run.run_id}  ({note})")
    print()
    while True:
        answer = input(f"Run to publish [1-{len(runs)}, q to quit]: ").strip().lower()
        if answer in {"q", "quit", ""}:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(runs):
            chosen = runs[int(answer) - 1]["run"]
            if not chosen.is_complete:
                print("That run covers only some calls, so it stays local. Pick another.")
                continue
            return chosen.run_id
        print("Enter one of the numbers above.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", help="Run directory name under evaluation/runs (omit to pick from a list)")
    args = parser.parse_args(argv)
    load_dotenv(REPOSITORY_ROOT / ".env")
    load_dotenv(EVALUATION_ROOT / ".env")
    started = time.perf_counter()
    try:
        publisher = SupabasePublisher.from_environment(progress=lambda message: print(f"  - {message}", flush=True))
        run_id = args.run_id
        if not run_id:
            if not sys.stdin.isatty():
                print("No --run-id given and no terminal to ask in.", file=sys.stderr)
                return 2
            run_id = choose_run(publisher)
            if not run_id:
                print("Nothing published.")
                return 0
        print(f"\nPublishing {run_id}")
        print("  - Validating run artifacts", flush=True)
        publication = validate_run_dir(EVALUATION_ROOT / "runs" / run_id, EVALUATION_ROOT)
        result = publisher.publish(publication)
    except PublicationError as error:
        print(f"Publication error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nCancelled. Re-running resumes safely; nothing is shown in the dashboard until it finishes.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Publication failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    if result == "already_published":
        print(f"\nRun {run_id} is already published with identical artifacts.")
    else:
        print(f"\nPublished run {run_id} in {elapsed:.0f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
