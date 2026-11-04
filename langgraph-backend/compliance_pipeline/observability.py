"""Local, metadata-only production measurements. Never sends data to a service.

The backend saves these measurements inside its existing AI output JSON. Prediction inputs, evidence and officer comments are excluded.
"""
from __future__ import annotations

import functools
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _enabled():
    return os.getenv("COMPLIANCE_OBSERVABILITY_ENABLED", "1") == "1"


def _iso(nanoseconds):
    return datetime.fromtimestamp(nanoseconds / 1e9, timezone.utc).isoformat()


def _tokens(models):
    result = {}
    if not isinstance(models, dict):
        return result
    for model, usage in (models or {}).items():
        if not re.fullmatch(r"[\w./:-]{1,200}", model) or not isinstance(usage, dict):
            continue
        def count(value):
            return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
        inputs, outputs = count(usage.get("input_tokens")), count(usage.get("output_tokens"))
        result[model] = {
            "input_tokens": inputs, "output_tokens": outputs, "total_tokens": inputs + outputs,
            "input_token_details": {"cache_read": min(inputs, count((usage.get("input_token_details") or {}).get("cache_read")))},
            "output_token_details": {"reasoning": min(outputs, count((usage.get("output_token_details") or {}).get("reasoning")))},
        }
    return result


def stage_record(name, started_ns, ended_ns, transports=None, calls=None, elapsed_ms=None):
    calls = calls or []
    return {
        "stage": name, "started_at": _iso(started_ns), "completed_at": _iso(ended_ns),
        "duration_ms": max(0.0, round(elapsed_ms if elapsed_ms is not None else (ended_ns - started_ns) / 1e6, 3)),
        "requests": len(calls), "attempts": sum(int(c.get("attempts") or 0) for c in calls),
        "status": "prediction_failure" if any(c.get("status") == "prediction_failure" for c in calls) else "success",
        "error_codes": sorted({m.group(1) for c in calls
                               if (m := re.search(r"\(code (\d+)\)", c.get("error") or ""))}),
        "usage_by_transport": {key: _tokens(value) for key, value in (transports or {}).items()
                               if key in {"standard", "batch"}},
    }


def track_stage(node):
    @functools.wraps(node)
    def wrapped(state, *args, **kwargs):
        started = time.time_ns()
        clock = time.perf_counter_ns()
        update = node(state, *args, **kwargs)
        if _enabled() and isinstance(update, dict):
            update = {**update, "stage_observations": [
                stage_record(node.__name__.removesuffix("_node"), started, time.time_ns(),
                             elapsed_ms=(time.perf_counter_ns() - clock) / 1e6)]}
        return update
    return wrapped


def begin_observation(state):
    if not _enabled():
        return {}
    try:
        identifier = str(uuid.UUID(str(state.get("analysis_run_id"))))
    except (ValueError, TypeError, AttributeError):
        identifier = str(uuid.uuid4())
    return {"analysis_run_id": identifier, "observation_started_ns": time.time_ns(),
            "observation_clock_ns": time.perf_counter_ns()}


def _cost(transports):
    from compliance_pipeline.usage import estimate_reported_cost
    default = Path(__file__).resolve().parent / "config" / "observability_prices.json"
    try:
        book = json.loads(Path(os.getenv("COMPLIANCE_PRICE_BOOK_PATH", str(default))).read_text(encoding="utf-8"))
        if datetime.now(timezone.utc).date().isoformat() > book["valid_until"]:
            return {"amount": None, "currency": "USD", "status": "price_snapshot_expired"}
        result = estimate_reported_cost(transports, book)
        return {**result, "status": "reported_usage_estimate" if result["amount"] is not None else "unavailable"}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {"amount": None, "currency": "USD", "status": "pricing_unavailable"}


def finish_observation(state):
    if not _enabled() or not state.get("observation_started_ns"):
        return {}
    ended = time.time_ns()
    started = state["observation_started_ns"]
    rows = [item.model_dump() if hasattr(item, "model_dump") else item for item in state.get("final_results", [])]
    failures = sum(item.get("execution_status") == "prediction_failure" for item in rows)
    status = "failed" if rows and failures == len(rows) else "partial" if failures else "completed"
    transports = {key: _tokens(value) for key, value in state.get("usage_by_transport", {}).items()
                  if key in {"standard", "batch"}}
    return {"observability": {
        "schema_version": "production-observability.v1",
        "analysis_run_id": state["analysis_run_id"],
        "trace_id": uuid.UUID(state["analysis_run_id"]).hex,
        "started_at": _iso(started), "completed_at": _iso(ended),
        "duration_ms": max(0.0, round((time.perf_counter_ns() - state.get("observation_clock_ns", time.perf_counter_ns())) / 1e6, 3)),
        "latency_scope": "graph processing through aggregation; writer and queue time excluded",
        "release": os.getenv("CLEARCALL_RELEASE", "unversioned")[:100],
        "status": status, "total_checks": len(rows), "prediction_failures": failures,
        "stages": state.get("stage_observations", []),
        "usage_by_transport": transports, "cost": _cost(state.get("usage_by_transport", {})),
    }}
