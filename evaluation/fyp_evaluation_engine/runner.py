"""Run the published dataset through the real graph once per complete call."""
from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from fyp_evaluation_loader import call_documents, graph_input_for_call, load_dataset

from . import ARTIFACT_SCHEMA_VERSION, METRIC_DEFINITIONS_VERSION, SCORER_VERSION
from .adapter import normalize_call_output
from .persistence import to_jsonable, write_run_artifacts
from .scoring import aggregate_scores, score_prediction

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _git_metadata() -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=REPOSITORY_ROOT, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = command("rev-parse", "HEAD")
    status = command("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("langgraph", "langchain", "langchain-aws", "pydantic"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _model_configuration() -> dict[str, Any]:
    return {
        "chat_model_id": os.getenv("CLEARCALL_MODEL_ID", "apac.amazon.nova-lite-v1:0"),
        "embedding_model_id": os.getenv("BEDROCK_EMBEDDING_MODEL", "cohere.embed-english-v3"),
        "aws_region": os.getenv("AWS_REGION", "ap-southeast-1"),
        "temperature": 0,
        "retrieval_top_k": int(os.getenv("CLEARCALL_TOP_K", "3")),
        "deterministic_top_k": int(os.getenv("CLEARCALL_DETERMINISTIC_TOP_K", "5")),
        "minimum_relevance": float(os.getenv("CLEARCALL_MIN_RELEVANCE", "0.25")),
        "max_review_chars": int(os.getenv("CLEARCALL_MAX_REVIEW_CHARS", "60000")),
        "package_versions": _package_versions(),
    }


def load_real_graph_invoke() -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Import the repository's actual compiled LangGraph lazily."""
    backend = REPOSITORY_ROOT / "langgraph-backend"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from compliance_pipeline.graph import graph  # noqa: PLC0415
    return graph.invoke


def normalize_and_score_call(
    call_id: str,
    checklist: list[dict[str, Any]],
    documents: dict[str, Any],
    output: Any,
    latency_ms: float,
    call_error: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize one call's graph output and score it. Shared by live runs and rescoring."""
    transcript_turn_ids = {segment["id"] for segment in documents["transcript"]["audio_segments"]}
    predictions, errors = normalize_call_output(
        call_id, checklist, transcript_turn_ids, output, latency_ms, call_error
    )
    gold_by_id = {check["question_id"]: check for check in documents["ground_truth"]["checks"]}
    scores = [score_prediction(prediction, gold_by_id[prediction["question_id"]]) for prediction in predictions]
    return predictions, errors, scores


def run_status(call_executions: list[dict[str, Any]], errors: list[dict[str, Any]]) -> str:
    graph_failures = sum(call["status"] != "success" for call in call_executions)
    return (
        "failed" if graph_failures == len(call_executions)
        else "completed_with_errors" if errors
        else "completed"
    )


def run_evaluation(
    dataset_dir: str | Path,
    output_root: str | Path,
    *,
    call_ids: list[str] | None = None,
    run_id: str | None = None,
    graph_invoke: Callable[[dict[str, Any]], Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute, normalize, score, and persist one immutable evaluation run."""
    dataset = load_dataset(dataset_dir)
    selected = list(call_ids or dataset["manifest"]["call_ids"])
    unknown = set(selected) - set(dataset["manifest"]["call_ids"])
    if unknown:
        raise ValueError(f"Unknown call IDs: {', '.join(sorted(unknown))}")
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select one or more unique call IDs")

    invoke = graph_invoke or load_real_graph_invoke()
    identifier = run_id or _new_run_id()
    if not identifier or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in identifier):
        raise ValueError("run_id may contain only letters, numbers, hyphens, and underscores")
    run_dir = Path(output_root).resolve() / identifier
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    started_at = _utc_now()
    run_started = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    raw_outputs: dict[str, Any] = {}
    call_executions: list[dict[str, Any]] = []
    observed_runtime_metadata: dict[str, dict[str, Any]] = {}
    checklist = dataset["checklist"]

    prior_bypass = os.environ.get("CLEARCALL_SKIP_APPLICABILITY")
    os.environ["CLEARCALL_SKIP_APPLICABILITY"] = "true"
    try:
        for call_id in selected:
            if progress:
                progress(f"{call_id}: invoking graph")
            documents = call_documents(dataset, call_id)
            graph_input = graph_input_for_call(dataset, call_id)
            call_started = time.perf_counter()
            output = None
            call_error = None
            exception_type = None
            try:
                # One invocation contains the full transcript and complete checklist.
                output = invoke(graph_input)
            except Exception as exc:  # persisted per call so the batch can finish
                exception_type = type(exc).__name__
                call_error = f"{exception_type}: {exc}"
            latency_ms = round((time.perf_counter() - call_started) * 1000, 3)
            if call_error is None:
                raw_outputs[call_id] = to_jsonable(output)
                if isinstance(output, dict):
                    observed = {
                        key: to_jsonable(output[key])
                        for key in ("usage_metadata", "estimated_cost", "trace_id")
                        if output.get(key) is not None
                    }
                    if observed:
                        observed_runtime_metadata[call_id] = observed
            else:
                raw_outputs[call_id] = {
                    "execution_status": "graph_exception",
                    "exception_type": exception_type,
                    "message": call_error,
                }
            call_predictions, call_errors, call_scores = normalize_and_score_call(
                call_id, checklist, documents, output, latency_ms, call_error
            )
            predictions.extend(call_predictions)
            errors.extend(call_errors)
            scores.extend(call_scores)
            call_executions.append({
                "call_id": call_id,
                "status": "success" if call_error is None else "graph_exception",
                "latency_ms": latency_ms,
                "question_count": len(call_predictions),
            })
            if progress:
                progress(f"{call_id}: {'completed' if call_error is None else 'failed'} in {latency_ms:.3f} ms")
    finally:
        if prior_bypass is None:
            os.environ.pop("CLEARCALL_SKIP_APPLICABILITY", None)
        else:
            os.environ["CLEARCALL_SKIP_APPLICABILITY"] = prior_bypass

    summary = aggregate_scores(scores)
    completed_at = _utc_now()
    total_latency_ms = round((time.perf_counter() - run_started) * 1000, 3)
    execution_status = run_status(call_executions, errors)
    git = _git_metadata()
    metadata = {
        "run_id": identifier,
        "run_kind": "graph_execution",
        "execution_status": execution_status,
        "started_at": started_at,
        "completed_at": completed_at,
        "total_latency_ms": total_latency_ms,
        "dataset_snapshot": {
            "path": str(dataset["root"]),
            "format": dataset["manifest"]["format"],
            "dataset_sha256": dataset["manifest"].get("dataset_sha256"),
            "manifest_sha256": dataset["manifest_sha256"],
            "source": dataset["manifest"].get("source"),
            "call_ids": selected,
        },
        "source_commit": dataset["manifest"].get("source", {}).get("git_commit"),
        "graph_commit": git["commit"],
        "graph_worktree_dirty": git["dirty"],
        "model_configuration": _model_configuration(),
        "execution_configuration": {
            "graph": "compliance_pipeline.graph:graph",
            "graph_invocations_per_call": 1,
            "applicability_bypassed": True,
            "applicability_environment": "CLEARCALL_SKIP_APPLICABILITY=true",
            "selected_call_ids": selected,
        },
        "scorer_version": SCORER_VERSION,
        "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
        "metric_definitions_path": str(REPOSITORY_ROOT / "evaluation" / "metric_definitions.json"),
        "call_executions": call_executions,
        "observability": (
            {"by_call_id": observed_runtime_metadata}
            if observed_runtime_metadata
            else {
                "token_usage": "not_exposed_by_current_graph_state",
                "estimated_cost": "not_exposed_by_current_graph_state",
                "trace_ids": "not_exposed_by_current_graph_state",
            }
        ),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
    }
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
