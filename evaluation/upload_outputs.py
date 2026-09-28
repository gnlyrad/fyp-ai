"""Upload compliance-pipeline outputs to MongoDB for evaluation.

Scans a directory (recursively) for JSON files written by the pipeline's
`JsonFileWriter` (`compliance_pipeline.nodes.output.build_payload` shape:
`schema_version`, `call_id`, `results`, ...) and upserts each one into
MongoDB. Files in any other shape (old response dumps, `score_report.json`,
...) are skipped.

If a run's `call_id` matches a fixture in
`langgraph-backend/compliance_pipeline/data/fixtures/<call_id>/expected.json`,
that expected result is stored alongside the run so the dashboard can score
it (accuracy, confusion matrix, ...).

Each document's `_id` is a SHA-256 of the file contents, so re-running the
upload is idempotent: an unchanged file is not duplicated, and a re-run of
the pipeline (new `generated_at`) is stored as a new run.

Usage:

    python evaluation/upload_outputs.py                       # langgraph-backend/output
    python evaluation/upload_outputs.py --dir path/to/output
    python evaluation/upload_outputs.py --tag gemini-baseline # label this batch
    python evaluation/upload_outputs.py --dry-run             # list what would be uploaded

Config (env vars or evaluation/.env):
    MONGODB_URI         default mongodb://localhost:27017
    MONGODB_DB          default compliance_eval
    MONGODB_COLLECTION  default pipeline_outputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "langgraph-backend" / "output"
FIXTURES_ROOT = REPO_ROOT / "langgraph-backend" / "compliance_pipeline" / "data" / "fixtures"

REQUIRED_KEYS = {"schema_version", "call_id", "results"}


def get_collection():
    from pymongo import ASCENDING, MongoClient

    load_dotenv(EVAL_DIR / ".env")
    client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    db = client[os.getenv("MONGODB_DB", "compliance_eval")]
    collection = db[os.getenv("MONGODB_COLLECTION", "pipeline_outputs")]
    collection.create_index([("call_id", ASCENDING)])
    collection.create_index([("tag", ASCENDING)])
    collection.create_index([("model.id", ASCENDING)])
    return collection


def load_expected(call_id: str) -> dict | None:
    path = FIXTURES_ROOT / call_id / "expected.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_document(path: Path, raw: bytes, payload: dict, tag: str | None) -> dict:
    call_id = payload.get("call_id", "unknown")
    return {
        **payload,
        "_id": hashlib.sha256(raw).hexdigest(),
        "tag": tag,
        "source_file": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "expected": load_expected(call_id),
    }


def collect(output_dir: Path) -> tuple[list[tuple[Path, bytes, dict]], list[tuple[Path, str]]]:
    found: list[tuple[Path, bytes, dict]] = []
    skipped: list[tuple[Path, str]] = []
    for path in sorted(output_dir.rglob("*.json")):
        raw = path.read_bytes()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            skipped.append((path, f"invalid JSON: {exc}"))
            continue
        if not isinstance(payload, dict) or not REQUIRED_KEYS <= payload.keys():
            skipped.append((path, "not a compliance-pipeline output"))
            continue
        found.append((path, raw, payload))
    return found, skipped


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory of pipeline outputs (searched recursively).")
    parser.add_argument("--tag", help="Label stored on every uploaded run, e.g. an experiment name.")
    parser.add_argument("--dry-run", action="store_true", help="List what would be uploaded without connecting to MongoDB.")
    args = parser.parse_args()

    output_dir = args.dir.resolve()
    if not output_dir.is_dir():
        print(f"Output directory not found: {output_dir}")
        return 1

    found, skipped = collect(output_dir)
    for path, reason in skipped:
        print(f"[skip]   {path.relative_to(output_dir)}: {reason}")

    if not found:
        print("No compliance-pipeline outputs found.")
        return 1

    collection = None if args.dry_run else get_collection()
    inserted = updated = 0
    for path, raw, payload in found:
        doc = build_document(path, raw, payload, args.tag)
        label = f"{path.relative_to(output_dir)} (call_id={doc['call_id']}, expected={'yes' if doc['expected'] else 'no'})"
        if collection is None:
            print(f"[would]  {label}")
            continue
        result = collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        if result.upserted_id is not None:
            inserted += 1
            print(f"[new]    {label}")
        else:
            updated += 1
            print(f"[update] {label}")

    print(f"\n{len(found)} output(s) found, {len(skipped)} skipped.")
    if collection is not None:
        print(f"{inserted} inserted, {updated} updated in {collection.database.name}.{collection.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
