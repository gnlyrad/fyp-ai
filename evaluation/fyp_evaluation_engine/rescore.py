"""Derive a new run from a saved run's raw graph outputs. Never invokes LangGraph or Bedrock."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fyp_evaluation_loader import call_documents, load_dataset

from . import ARTIFACT_SCHEMA_VERSION, METRIC_DEFINITIONS_VERSION, SCORER_VERSION
from .persistence import write_run_artifacts
from .runner import REPOSITORY_ROOT, _new_run_id, _utc_now, normalize_and_score_call, run_status
from .scoring import aggregate_scores


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rescore_run(
    source_run_dir: str | Path,
    output_root: str | Path,
    dataset_dir: str | Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Re-normalize, rescore, and re-aggregate ``raw/<call_id>.json`` into a NEW run.

    The source run is only read. The dataset must be byte-identical (by manifest and
    dataset fingerprint) to the one the source run used, otherwise this refuses.
    """
    source = Path(source_run_dir).resolve()
    source_metadata = _read_json(source / "metadata.json")
    dataset = load_dataset(dataset_dir)
    snapshot = source_metadata["dataset_snapshot"]
    for key, actual in (
        ("dataset_sha256", dataset["manifest"].get("dataset_sha256")),
        ("manifest_sha256", dataset["manifest_sha256"]),
    ):
        if snapshot.get(key) != actual:
            raise ValueError(f"Dataset {key} differs from source run {source.name}; refusing to rescore")

    identifier = run_id or _new_run_id()
    run_dir = Path(output_root).resolve() / identifier
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    latency_by_call = {call["call_id"]: call for call in source_metadata["call_executions"]}
    checklist = dataset["checklist"]
    predictions: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    raw_outputs: dict[str, Any] = {}
    call_executions: list[dict[str, Any]] = []
    for call_id in snapshot["call_ids"]:
        raw = _read_json(source / "raw" / f"{call_id}.json")
        raw_outputs[call_id] = raw
        failed = isinstance(raw, dict) and raw.get("execution_status") == "graph_exception"
        latency_ms = latency_by_call[call_id]["latency_ms"]
        call_predictions, call_errors, call_scores = normalize_and_score_call(
            call_id, checklist, call_documents(dataset, call_id),
            None if failed else raw, latency_ms, raw.get("message") if failed else None,
        )
        predictions.extend(call_predictions)
        errors.extend(call_errors)
        scores.extend(call_scores)
        call_executions.append({
            "call_id": call_id,
            "status": "graph_exception" if failed else "success",
            "latency_ms": latency_ms,
            "question_count": len(call_predictions),
        })

    source_kind = source_metadata.get("run_kind", "graph_execution")
    metadata = {
        **source_metadata,
        "run_id": identifier,
        "run_kind": "rescored",
        "execution_status": run_status(call_executions, errors),
        "rescored_at": _utc_now(),
        "dataset_snapshot": {**snapshot, "path": str(dataset["root"])},
        "scorer_version": SCORER_VERSION,
        "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
        "metric_definitions_path": str(REPOSITORY_ROOT / "evaluation" / "metric_definitions.json"),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "call_executions": call_executions,
        "derived_from": {
            "run_id": source_metadata["run_id"],
            "run_kind": source_kind,
            "scorer_version": source_metadata.get("scorer_version"),
            "metric_definitions_version": source_metadata.get("metric_definitions_version"),
            "artifact_schema_version": source_metadata.get("artifact_schema_version"),
            "dataset_sha256": snapshot.get("dataset_sha256"),
            "manifest_sha256": snapshot.get("manifest_sha256"),
            "graph_commit": source_metadata.get("graph_commit"),
            "graph_invoked": False,
            "method": "raw/<call_id>.json re-normalized, rescored, and re-aggregated",
        },
    }
    summary = aggregate_scores(scores)
    write_run_artifacts(
        run_dir,
        metadata=metadata,
        raw_outputs=raw_outputs,
        predictions=predictions,
        scores=scores,
        summary=summary,
        errors=errors,
    )
    return {"run_dir": run_dir, "metadata": metadata, "summary": summary, "errors": errors}
