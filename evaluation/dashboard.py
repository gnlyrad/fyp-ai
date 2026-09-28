"""Evaluation dashboard for compliance-pipeline outputs stored in MongoDB.

Pulls every run uploaded by `upload_outputs.py`, computes evaluation metrics,
and serves them to a small web UI (`evaluation/static/index.html`).

Two kinds of metrics:
- Operational (every run): status distribution, review/escalation rates,
  confidence, verification, warnings, red flags, errors, quality flags.
- Accuracy (only runs whose `call_id` matched a fixture, so an `expected`
  result was stored at upload): per-item and overall status accuracy,
  call-intent / escalation accuracy, action-type and red-flag recall, and a
  status confusion matrix. Same rules as `langgraph-backend/scripts/
  score_fixtures.py`: an item matches when the actual status is one of the
  expected acceptable statuses.

Usage:

    python evaluation/dashboard.py              # http://localhost:8050
    python evaluation/dashboard.py --port 9000

Config (env vars or evaluation/.env): MONGODB_URI, MONGODB_DB,
MONGODB_COLLECTION (same defaults as upload_outputs.py).
"""
from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pymongo import DESCENDING, MongoClient

EVAL_DIR = Path(__file__).resolve().parent
STATIC_DIR = EVAL_DIR / "static"
STATUSES = ["PASS", "PARTIAL", "FAIL", "NOT_APPLICABLE", "CANNOT_DETERMINE"]

load_dotenv(EVAL_DIR / ".env")
_client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
collection = _client[os.getenv("MONGODB_DB", "compliance_eval")][os.getenv("MONGODB_COLLECTION", "pipeline_outputs")]

app = FastAPI(title="Compliance pipeline evaluation")


# ---------------------------------------------------------------------------
# Data access.
# ---------------------------------------------------------------------------

def _query(tag: str | None, model: str | None, call_id: str | None) -> dict:
    q: dict = {}
    if tag:
        q["tag"] = None if tag == "(none)" else tag
    if model:
        q["model.id"] = model
    if call_id:
        q["call_id"] = call_id
    return q


def load_runs(tag: str | None, model: str | None, call_id: str | None) -> list[dict]:
    projection = {"transcript": 0, "segments": 0, "rules": 0}
    return list(collection.find(_query(tag, model, call_id), projection).sort("generated_at", DESCENDING))


# ---------------------------------------------------------------------------
# Scoring (pure).
# ---------------------------------------------------------------------------

def _acceptable(value) -> list[str]:
    return [value] if isinstance(value, str) else list(value or [])


def score_run(run: dict) -> dict | None:
    """Compare one run against its stored `expected`; None when there is none."""
    expected = run.get("expected")
    if not expected:
        return None
    actual_by_id = {r.get("item_id"): r.get("status") for r in run.get("results") or []}
    items = {}
    for item_id, value in (expected.get("items") or {}).items():
        acceptable = _acceptable(value)
        actual = actual_by_id.get(item_id)
        items[item_id] = {"expected": acceptable, "actual": actual, "match": actual in acceptable}

    actual_intent = (run.get("call_intent") or {}).get("category")
    actual_escalation = bool((run.get("escalation") or {}).get("required", False))
    expected_actions = set(expected.get("expected_action_types") or [])
    actual_actions = {a.get("action_type") for a in run.get("actions") or []} - {None}
    expected_flags = set(expected.get("expected_red_flag_categories") or [])
    actual_flags = {f.get("category") for f in run.get("red_flags") or []} - {None}

    return {
        "items": items,
        "call_intent": {"expected": expected.get("call_intent"), "actual": actual_intent,
                        "match": actual_intent == expected.get("call_intent")},
        "escalation": {"expected": bool(expected.get("escalation_required", False)), "actual": actual_escalation,
                       "match": actual_escalation == bool(expected.get("escalation_required", False))},
        "action_types": {"expected": sorted(expected_actions), "actual": sorted(actual_actions),
                         "missing": sorted(expected_actions - actual_actions),
                         "unexpected": sorted(actual_actions - expected_actions),
                         "match": expected_actions <= actual_actions},
        "red_flag_categories": {"expected": sorted(expected_flags), "actual": sorted(actual_flags),
                                "missing": sorted(expected_flags - actual_flags),
                                "unexpected": sorted(actual_flags - expected_flags),
                                "match": expected_flags <= actual_flags},
    }


def _rate(hits: int, total: int) -> dict:
    return {"hits": hits, "total": total, "rate": hits / total if total else None}


def compute_metrics(runs: list[dict]) -> dict:
    n = len(runs)
    status_totals: Counter = Counter()
    item_status: dict[str, Counter] = defaultdict(Counter)
    item_conf: dict[str, list[float]] = defaultdict(list)
    item_names: dict[str, str] = {}
    quality_flags: Counter = Counter()
    red_flag_severity: Counter = Counter()
    red_flag_category: Counter = Counter()
    actions_by_owner: Counter = Counter()
    intents: Counter = Counter()
    models: Counter = Counter()
    needs_review = escalated = held = runs_with_errors = 0
    warnings = verified = item_count = 0

    for run in runs:
        summary = run.get("summary") or {}
        escalation = run.get("escalation") or {}
        needs_review += bool(summary.get("needs_review"))
        escalated += bool(escalation.get("required"))
        held += bool(escalation.get("hold_transaction"))
        runs_with_errors += bool(run.get("errors"))
        quality_flags.update(run.get("quality_flags") or [])
        intents[(run.get("call_intent") or {}).get("category") or "undetermined"] += 1
        models[(run.get("model") or {}).get("id") or "unknown"] += 1
        for flag in run.get("red_flags") or []:
            red_flag_severity[flag.get("severity") or "?"] += 1
            red_flag_category[flag.get("category") or "?"] += 1
        for action in run.get("actions") or []:
            actions_by_owner[action.get("owner_role") or "?"] += 1
        for item in run.get("checklist") or []:
            item_names.setdefault(item.get("id"), item.get("name"))
        for result in run.get("results") or []:
            item_id = result.get("item_id", "?")
            status = result.get("status", "?")
            item_count += 1
            status_totals[status] += 1
            item_status[item_id][status] += 1
            if isinstance(result.get("confidence"), (int, float)):
                item_conf[item_id].append(float(result["confidence"]))
            verified += bool(result.get("verified"))
            warnings += len(result.get("warnings") or [])

    all_conf = [c for values in item_conf.values() for c in values]
    per_item = {
        item_id: {
            "statuses": dict(counts),
            "total": sum(counts.values()),
            "avg_confidence": sum(item_conf[item_id]) / len(item_conf[item_id]) if item_conf[item_id] else None,
        }
        for item_id, counts in sorted(item_status.items())
    }

    operational = {
        "runs": n,
        "needs_review": _rate(needs_review, n),
        "escalated": _rate(escalated, n),
        "hold_transaction": _rate(held, n),
        "runs_with_errors": _rate(runs_with_errors, n),
        "verified_items": _rate(verified, item_count),
        "avg_confidence": sum(all_conf) / len(all_conf) if all_conf else None,
        "avg_warnings_per_run": warnings / n if n else None,
        "status_totals": {s: status_totals.get(s, 0) for s in STATUSES},
        "per_item": per_item,
        "quality_flags": dict(quality_flags.most_common()),
        "call_intents": dict(intents.most_common()),
        "models": dict(models.most_common()),
        "red_flags_by_severity": dict(red_flag_severity.most_common()),
        "red_flags_by_category": dict(red_flag_category.most_common()),
        "actions_by_owner": dict(actions_by_owner.most_common()),
    }
    return {"operational": operational, "accuracy": compute_accuracy(runs), "item_names": item_names}


def compute_accuracy(runs: list[dict]) -> dict | None:
    scored = [(run, score) for run in runs if (score := score_run(run)) is not None]
    if not scored:
        return None
    item_hits: Counter = Counter()
    item_totals: Counter = Counter()
    confusion: dict[str, Counter] = defaultdict(Counter)
    field_hits: Counter = Counter()
    per_run = []

    for run, score in scored:
        hits = 0
        for item_id, s in score["items"].items():
            item_totals[item_id] += 1
            item_hits[item_id] += s["match"]
            hits += s["match"]
            confusion[s["expected"][0] if s["expected"] else "?"][s["actual"] or "?"] += 1
        for field in ("call_intent", "escalation", "action_types", "red_flag_categories"):
            field_hits[field] += score[field]["match"]
        per_run.append({
            "_id": run["_id"], "call_id": run.get("call_id"), "tag": run.get("tag"),
            "model": (run.get("model") or {}).get("id"), "generated_at": run.get("generated_at"),
            "item_accuracy": _rate(hits, len(score["items"])), "score": score,
        })

    n = len(scored)
    return {
        "scored_runs": n,
        "overall_item_accuracy": _rate(sum(item_hits.values()), sum(item_totals.values())),
        "per_item_accuracy": {i: _rate(item_hits[i], item_totals[i]) for i in sorted(item_totals)},
        "call_intent_accuracy": _rate(field_hits["call_intent"], n),
        "escalation_accuracy": _rate(field_hits["escalation"], n),
        "action_type_recall": _rate(field_hits["action_types"], n),
        "red_flag_recall": _rate(field_hits["red_flag_categories"], n),
        "confusion_matrix": {"labels": STATUSES + ["?"], "matrix": {k: dict(v) for k, v in confusion.items()}},
        "per_run": per_run,
    }


def run_row(run: dict) -> dict:
    summary = run.get("summary") or {}
    score = score_run(run)
    item_acc = None
    if score:
        items = score["items"].values()
        item_acc = _rate(sum(s["match"] for s in items), len(score["items"]))
    return {
        "_id": run["_id"],
        "call_id": run.get("call_id"),
        "tag": run.get("tag"),
        "model": (run.get("model") or {}).get("id"),
        "generated_at": run.get("generated_at"),
        "call_intent": (run.get("call_intent") or {}).get("category"),
        "counts": summary.get("counts") or {},
        "needs_review": summary.get("needs_review"),
        "escalated": (run.get("escalation") or {}).get("required"),
        "red_flags": len(run.get("red_flags") or []),
        "errors": len(run.get("errors") or []),
        "item_accuracy": item_acc,
    }


# ---------------------------------------------------------------------------
# API.
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/filters")
def filters() -> dict:
    tags = collection.distinct("tag")
    return {
        "tags": sorted(t if t is not None else "(none)" for t in tags),
        "models": sorted(m for m in collection.distinct("model.id") if m),
        "call_ids": sorted(c for c in collection.distinct("call_id") if c),
    }


@app.get("/api/metrics")
def metrics(tag: str | None = Query(None), model: str | None = Query(None), call_id: str | None = Query(None)) -> dict:
    return compute_metrics(load_runs(tag, model, call_id))


@app.get("/api/runs")
def runs(tag: str | None = Query(None), model: str | None = Query(None), call_id: str | None = Query(None)) -> list[dict]:
    return [run_row(r) for r in load_runs(tag, model, call_id)]


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict:
    run = collection.find_one({"_id": run_id})
    if run is None:
        raise HTTPException(404, "run not found")
    run["score"] = score_run(run)
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
