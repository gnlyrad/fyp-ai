"""Evaluation dashboard for scored compliance-pipeline outputs in MongoDB.

Serves the outputs uploaded (and scored) by `upload_outputs.py` to a small
web UI (`evaluation/static/index.html`). Per-output scores are stored at
upload; this service only filters runs and aggregates them with
`scoring.aggregate`, so every view uses the same formulas.

Two kinds of metrics:
- Evaluation (outputs whose call_id matched a fixture): status accuracy,
  breach detection, call intent, escalation, action types, red flags.
- Operational (every output): review/escalation rates, confidence,
  verification, warnings, errors, status distribution.

By default only the latest run of each call is included, so re-running the
pipeline does not mix old and new results; untick "Latest run per call" to
include every uploaded run.

Usage:

    python evaluation/dashboard.py              # http://127.0.0.1:8050
    python evaluation/dashboard.py --port 9000

Config (env vars or evaluation/.env): MONGODB_URI, MONGODB_DB,
MONGODB_COLLECTION (same defaults as upload_outputs.py).
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pymongo import DESCENDING, MongoClient

from scoring import METRIC_DEFINITIONS, SCORER_VERSION, STATUSES, aggregate, ratio

EVAL_DIR = Path(__file__).resolve().parent
STATIC_DIR = EVAL_DIR / "static"

load_dotenv(EVAL_DIR / ".env")
collection = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))[
    os.getenv("MONGODB_DB", "compliance_eval")][os.getenv("MONGODB_COLLECTION", "pipeline_outputs")]

app = FastAPI(title="Compliance pipeline evaluation")


def load_runs(model: str | None, call_id: str | None, latest: bool) -> list[dict]:
    q: dict = {}
    if model:
        q["model.id"] = model
    if call_id:
        q["call_id"] = call_id
    projection = {"transcript": 0, "segments": 0, "rules": 0}
    runs = list(collection.find(q, projection).sort("generated_at", DESCENDING))
    if latest:
        seen: set[str] = set()
        runs = [r for r in runs if not (r.get("call_id") in seen or seen.add(r.get("call_id")))]
    return runs


def operational(runs: list[dict]) -> dict:
    n = len(runs)
    status_totals: Counter = Counter()
    quality_flags: Counter = Counter()
    intents: Counter = Counter()
    models: Counter = Counter()
    owners: Counter = Counter()
    severities: Counter = Counter()
    confidences: list[float] = []
    verified = items = warnings = 0
    for run in runs:
        quality_flags.update(run.get("quality_flags") or [])
        intents[(run.get("call_intent") or {}).get("category") or "undetermined"] += 1
        models[(run.get("model") or {}).get("id") or "unknown"] += 1
        owners.update(a.get("owner_role") or "?" for a in run.get("actions") or [])
        severities.update(f.get("severity") or "?" for f in run.get("red_flags") or [])
        for r in run.get("results") or []:
            items += 1
            status_totals[r.get("status", "?")] += 1
            verified += bool(r.get("verified"))
            warnings += len(r.get("warnings") or [])
            if isinstance(r.get("confidence"), (int, float)):
                confidences.append(float(r["confidence"]))
    return {
        "runs": n,
        "needs_review": ratio(sum(bool((r.get("summary") or {}).get("needs_review")) for r in runs), n),
        "escalated": ratio(sum(bool((r.get("escalation") or {}).get("required")) for r in runs), n),
        "hold_transaction": ratio(sum(bool((r.get("escalation") or {}).get("hold_transaction")) for r in runs), n),
        "runs_with_errors": ratio(sum(bool(r.get("errors")) for r in runs), n),
        "verified_items": ratio(verified, items),
        "avg_confidence": sum(confidences) / len(confidences) if confidences else None,
        "avg_warnings_per_run": warnings / n if n else None,
        "status_totals": {s: status_totals.get(s, 0) for s in STATUSES},
        "quality_flags": dict(quality_flags.most_common()),
        "call_intents": dict(intents.most_common()),
        "models": dict(models.most_common()),
        "actions_by_owner": dict(owners.most_common()),
        "red_flags_by_severity": dict(severities.most_common()),
    }


def item_names(runs: list[dict]) -> dict[str, str]:
    names: dict[str, str] = {}
    for run in runs:
        for item in run.get("checklist") or []:
            names.setdefault(item.get("id"), item.get("name"))
    return names


def run_row(run: dict) -> dict:
    score = run.get("score")
    return {
        "_id": run["_id"],
        "call_id": run.get("call_id"),
        "model": (run.get("model") or {}).get("id"),
        "generated_at": run.get("generated_at"),
        "source_file": run.get("source_file"),
        "call_intent": (run.get("call_intent") or {}).get("category"),
        "counts": (run.get("summary") or {}).get("counts") or {},
        "needs_review": (run.get("summary") or {}).get("needs_review"),
        "escalated": (run.get("escalation") or {}).get("required"),
        "errors": len(run.get("errors") or []),
        "score": None if score is None else {
            "item_accuracy": score["item_accuracy"],
            "call_intent": score["call_intent"]["match"],
            "escalation": score["escalation"]["match"],
            "action_types": score["action_types"]["match"],
            "red_flag_categories": score["red_flag_categories"]["match"],
            "breaches_missed": sum(i["breach_outcome"] not in (None, "detected") for i in score["items"]),
        },
    }


# ---------------------------------------------------------------------------
# API.
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/definitions")
def definitions() -> dict:
    return {"scorer_version": SCORER_VERSION, "metrics": METRIC_DEFINITIONS}


@app.get("/api/filters")
def filters() -> dict:
    return {
        "models": sorted(m for m in collection.distinct("model.id") if m),
        "call_ids": sorted(c for c in collection.distinct("call_id") if c),
    }


@app.get("/api/metrics")
def metrics(model: str | None = Query(None), call_id: str | None = Query(None), latest: bool = Query(True)) -> dict:
    runs = load_runs(model, call_id, latest)
    scored = [r for r in runs if r.get("score")]
    return {
        "operational": operational(runs),
        "evaluation": aggregate(scored),
        "scored_runs": len(scored),
        "unscored_runs": len(runs) - len(scored),
        "item_names": item_names(runs),
    }


@app.get("/api/runs")
def runs(model: str | None = Query(None), call_id: str | None = Query(None), latest: bool = Query(True)) -> list[dict]:
    return [run_row(r) for r in load_runs(model, call_id, latest)]


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    run = collection.find_one({"_id": run_id})
    if run is None:
        raise HTTPException(404, "run not found")
    return run


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    print(f"Evaluation dashboard: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
