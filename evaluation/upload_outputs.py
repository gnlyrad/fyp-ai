"""Score compliance-pipeline outputs and upload them to MongoDB.

Scans `langgraph-backend/output/` (recursively) for JSON files written by the
pipeline (`compliance_pipeline.nodes.output.build_payload` shape:
`schema_version`, `call_id`, `results`, ...). Other files (old response
dumps, `score_report.json`, ...) are skipped.

Each output whose `call_id` matches a fixture in
`langgraph-backend/compliance_pipeline/data/fixtures/<call_id>/expected.json`
is scored against it (see `scoring.py`); the expected result and the score
are stored with the output. Outputs with no matching fixture are uploaded
unscored and only count toward the dashboard's operational metrics.

Each document's `_id` is a SHA-256 of the file contents, so re-running the
upload never duplicates an unchanged file (it is re-scored in place), and a
new pipeline run of the same call is stored as a new run.

Usage:

    python evaluation/upload_outputs.py              # everything in langgraph-backend/output
    python evaluation/upload_outputs.py --dir path/to/output
    python evaluation/upload_outputs.py --dry-run    # score and list, without MongoDB

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

from scoring import score_output

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
    collection.create_index([("call_id", ASCENDING), ("generated_at", ASCENDING)])
    collection.create_index([("model.id", ASCENDING)])
    return collection


def load_expected(call_id: str) -> dict | None:
    path = FIXTURES_ROOT / call_id / "expected.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_document(path: Path, raw: bytes, payload: dict) -> dict:
    expected = load_expected(payload.get("call_id", "unknown"))
    return {
        **payload,
        "_id": hashlib.sha256(raw).hexdigest(),
        "source_file": str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "expected": expected,
        "score": score_output(expected, payload) if expected else None,
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
    parser.add_argument("--dry-run", action="store_true", help="Score and list outputs without connecting to MongoDB.")
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
    inserted = updated = scored = 0
    for path, raw, payload in found:
        doc = build_document(path, raw, payload)
        score = doc["score"]
        scored += score is not None
        result = (f"items {score['item_accuracy']['numerator']}/{score['item_accuracy']['denominator']}"
                  if score else "no fixture, unscored")
        label = f"{path.relative_to(output_dir)} (call_id={doc['call_id']}, {result})"
        if collection is None:
            print(f"[would]  {label}")
            continue
        existed = collection.count_documents({"_id": doc["_id"]}, limit=1)
        collection.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        if existed:
            updated += 1
        else:
            inserted += 1
        print(f"[{'update' if existed else 'new':<6}] {label}")

    print(f"\n{len(found)} output(s) found ({scored} scored against fixtures), {len(skipped)} skipped.")
    if collection is not None:
        print(f"{inserted} inserted, {updated} updated in {collection.database.name}.{collection.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
