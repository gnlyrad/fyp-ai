"""Run the published dataset through the real graph once per complete call."""
from __future__ import annotations

import importlib.metadata
import hashlib
import os
import re
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
from .current_pipeline import prepare_input, adapt_output
from .persistence import to_jsonable, write_run_artifacts
from .scoring import aggregate_scores, score_prediction
from .usage import PRICING_FILE, invoke_with_usage, load_pricing, model_key, resolve_pricing, summarize_run_usage

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
                ["git", "-c", f"safe.directory={REPOSITORY_ROOT.as_posix()}", *args], cwd=REPOSITORY_ROOT, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = command("rev-parse", "HEAD")
    status = command("status", "--porcelain")
    author = command("show", "-s", "--format=%an", "HEAD") if commit else None
    committed_at = command("show", "-s", "--format=%cI", "HEAD") if commit else None
    return {
        "commit": commit,
        "author": author,
        "committed_at": committed_at,
        "dirty": bool(status) if status is not None else None,
    }


def _git_commit_identity(commit: str | None) -> dict[str, str | None]:
    """Resolve immutable author metadata for an already-recorded commit."""
    if not commit:
        return {"author": None, "committed_at": None}

    def command(format_value: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-c", f"safe.directory={REPOSITORY_ROOT.as_posix()}", "show", "-s", format_value, commit],
                cwd=REPOSITORY_ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {"author": command("--format=%an"), "committed_at": command("--format=%cI")}


def _sha256_file(path: Path) -> str:
    """Hash a versioned input file exactly as persisted on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("langgraph", "langchain", "langchain-google-genai", "pydantic"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _model_configuration() -> dict[str, Any]:
    return {
        "model_provider": "google_genai",
        "chat_model_id": os.getenv("COMPLIANCE_MODEL_ID") or os.getenv("CLEARCALL_MODEL_ID", "gemini-3.5-flash"),
        "embedding_model_id": os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001"),
        "temperature": float(os.getenv("CLEARCALL_TEMPERATURE", "1")),
        "structured_method": os.getenv("COMPLIANCE_STRUCTURED_METHOD", "function_calling"),
        "sampling_seed": os.getenv("COMPLIANCE_LLM_SEED", "42"),
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
    def invoke(value):
        if value.get("context_mode") != "full_transcript":
            raise ValueError("The staged pipeline supports full_transcript only")
        return invoke_with_usage(graph, prepare_input(value))
    return invoke


# The graph's errors channel is list[str]. Its per-question nodes prefix every
# message with "<node>[<checklist id>]:", and checklist IDs pass through the
# adapter unchanged, so the prefix is the graph's own attribution.
_QUESTION_DIAGNOSTIC = re.compile(r"^(?:check_item|verify)\[([^\]]+)\]:")


def diagnostic_question_id(message: Any, question_ids: set[str]) -> str | None:
    """Attribute a graph diagnostic to a checklist question; None when it is not question-level."""
    match = _QUESTION_DIAGNOSTIC.match(message) if isinstance(message, str) else None
    return match.group(1) if match and match.group(1) in question_ids else None


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
    native = isinstance(output, dict) and "final_results" in output
    graph_errors = list(output.get("errors", [])) if native else []
    if native:
        output = adapt_output(output, documents["transcript"])
    predictions, errors = normalize_call_output(
        call_id, checklist, transcript_turn_ids, output, latency_ms, call_error
    )
    question_ids = {item["id"] for item in checklist}
    errors.extend({"category": "pipeline_diagnostic", "call_id": call_id,
                   "question_id": diagnostic_question_id(message, question_ids), "message": message}
                  for message in graph_errors)
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
    context_mode: str | None = None,
    analysis_max_attempts: int | None = None,
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
    # Resolve after graph import, which loads the configured environment file.
    mode = context_mode if context_mode is not None else os.getenv("CLEARCALL_CONTEXT_MODE", "full_transcript")
    attempts = (analysis_max_attempts if analysis_max_attempts is not None
                else os.getenv("CLEARCALL_ANALYSIS_MAX_ATTEMPTS", "3"))
    if mode not in {"routed", "full_transcript"}:
        raise ValueError("context_mode must be routed or full_transcript")
    if isinstance(attempts, bool) or str(attempts) not in {"1", "2", "3", "4", "5"}:
        raise ValueError("analysis_max_attempts must be an integer from 1 to 5")
    if graph_invoke is None and mode != "full_transcript":
        raise ValueError("The staged pipeline supports full_transcript only")
    analysis_configuration = {"context_mode": mode, "analysis_max_attempts": int(attempts)}
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
    usage_by_call: dict[str, dict[str, Any]] = {}
    checklist = dataset["checklist"]

    prior_attempts = os.environ.get("COMPLIANCE_LLM_MAX_ATTEMPTS")
    os.environ["COMPLIANCE_LLM_MAX_ATTEMPTS"] = str(attempts)
    try:
        for call_id in selected:
            if progress:
                progress(f"{call_id}: invoking graph")
            documents = call_documents(dataset, call_id)
            graph_input = graph_input_for_call(dataset, call_id)
            graph_input.update(analysis_configuration)
            graph_input["call_id"] = call_id
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
                partial_usage = getattr(exc, "usage_metadata", None)
                if isinstance(partial_usage, dict) and partial_usage:
                    usage_by_call[call_id] = to_jsonable(partial_usage)
            latency_ms = round((time.perf_counter() - call_started) * 1000, 3)
            if call_error is None:
                raw_outputs[call_id] = to_jsonable(output)
                if isinstance(output, dict):
                    # The pipeline reports its own `usage`; the outer callback total is the fallback
                    # for graphs that do not.
                    reported = output.get("usage") or output.get("usage_metadata")
                    if isinstance(reported, dict) and reported:
                        usage_by_call[call_id] = to_jsonable(reported)
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
        if prior_attempts is None:
            os.environ.pop("COMPLIANCE_LLM_MAX_ATTEMPTS", None)
        else:
            os.environ["COMPLIANCE_LLM_MAX_ATTEMPTS"] = prior_attempts

    summary = aggregate_scores(scores)
    reported_models = {model_key(model) for usage in usage_by_call.values() for model in usage}
    run_pricing = resolve_pricing(load_pricing(REPOSITORY_ROOT / "evaluation" / PRICING_FILE), reported_models)
    completed_at = _utc_now()
    total_latency_ms = round((time.perf_counter() - run_started) * 1000, 3)
    execution_status = run_status(call_executions, errors)
    git = _git_metadata()
    metric_definitions_path = REPOSITORY_ROOT / "evaluation" / "metric_definitions.json"
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
            "checklist_sha256": dataset["manifest"].get("files", {}).get("checklist.json"),
            "source": dataset["manifest"].get("source"),
            "call_ids": selected,
        },
        "source_commit": dataset["manifest"].get("source", {}).get("git_commit"),
        "graph_commit": git["commit"],
        "git_commit_author": git["author"],
        "git_commit_timestamp": git["committed_at"],
        "graph_worktree_dirty": git["dirty"],
        "model_configuration": _model_configuration(),
        "execution_configuration": {
            **analysis_configuration,
            "graph": "compliance_pipeline.graph:graph",
            "graph_invocations_per_call": 1,
            "applicability_bypassed": True,
            "applicability_environment": "all stage-2 items run; gate is not wired",
            "pipeline_format": "staged-v1" if graph_invoke is None else "injected",
            "llm_max_attempts": int(attempts),
            "selected_call_ids": selected,
        },
        "scorer_version": SCORER_VERSION,
        "metric_definitions_version": METRIC_DEFINITIONS_VERSION,
        "metric_definitions_sha256": _sha256_file(metric_definitions_path),
        "metric_definitions_path": str(metric_definitions_path),
        "call_executions": call_executions,
        "observability": summarize_run_usage(usage_by_call, run_pricing),
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
