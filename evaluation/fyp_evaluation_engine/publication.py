"""Validate and publish one immutable evaluator run to Supabase."""
from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fyp_evaluation_loader import call_documents, load_dataset

from dashboard.view_models import metric_index, nested

REQUIRED_ARTIFACTS = (
    "metadata.json",
    "summary.json",
    "results.csv",
    "scores.jsonl",
    "predictions.jsonl",
    "errors.jsonl",
)
BUCKET = "evaluation-artifacts"
INSERT_CHUNK = 100
UPLOAD_WORKERS = 8


class PublicationError(RuntimeError):
    """A run is incomplete, incompatible, unauthorized, or collides with cloud state."""


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationError(f"Cannot read {path}: {error}") from error


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise PublicationError(f"Cannot read {path}: {error}") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def directory_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@dataclass(frozen=True)
class ValidatedPublication:
    evaluation_root: Path
    run_dir: Path
    dataset: dict[str, Any]
    metadata: dict[str, Any]
    summary: dict[str, Any]
    results: list[dict[str, Any]]
    scores: list[dict[str, Any]]
    predictions: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    artifact_fingerprint: str


def _find_dataset(evaluation_root: Path, dataset_sha256: str) -> dict[str, Any]:
    for path in (evaluation_root / "datasets").iterdir():
        if not path.is_dir():
            continue
        try:
            dataset = load_dataset(path)
        except Exception:
            continue
        if dataset["manifest"].get("dataset_sha256") == dataset_sha256:
            return dataset
    raise PublicationError(f"Local dataset snapshot not found for {dataset_sha256}")


def validate_run_dir(run_dir: str | Path, evaluation_root: str | Path) -> ValidatedPublication:
    run_dir = Path(run_dir).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    if not run_dir.is_dir():
        raise PublicationError(f"Run directory not found: {run_dir}")
    for name in REQUIRED_ARTIFACTS:
        if not (run_dir / name).is_file():
            raise PublicationError(f"Required run artifact is missing: {name}")
    metadata = _json(run_dir / "metadata.json")
    if metadata.get("run_id") != run_dir.name:
        raise PublicationError("metadata.run_id does not match the run directory")
    if metadata.get("execution_status") not in {"completed", "completed_with_errors"}:
        raise PublicationError("Only completed or completed_with_errors runs may be published")
    snapshot = metadata.get("dataset_snapshot") or {}
    dataset = _find_dataset(evaluation_root, snapshot.get("dataset_sha256"))
    manifest = dataset["manifest"]
    if snapshot.get("manifest_sha256") != dataset["manifest_sha256"]:
        raise PublicationError("Run manifest fingerprint does not match the local dataset")
    if snapshot.get("checklist_sha256") != manifest.get("files", {}).get("checklist.json"):
        raise PublicationError("Run checklist fingerprint is missing or does not match the dataset")
    selected = metadata.get("execution_configuration", {}).get("selected_call_ids") or snapshot.get("call_ids") or []
    if set(selected) != set(manifest["call_ids"]) or len(selected) != len(manifest["call_ids"]):
        raise PublicationError("Partial-call runs remain local and cannot be published")
    if not metadata.get("metric_definitions_sha256"):
        raise PublicationError("Run metadata is missing metric_definitions_sha256; rescore or rerun it first")
    for field in ("artifact_schema_version", "scorer_version", "metric_definitions_version"):
        if not metadata.get(field):
            raise PublicationError(f"Run metadata is missing {field}")
    raw_dir = run_dir / "raw"
    for call_id in manifest["call_ids"]:
        if not (raw_dir / f"{call_id}.json").is_file():
            raise PublicationError(f"Raw pipeline output is missing for {call_id}")
    with (run_dir / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        results = list(csv.DictReader(handle))
    scores = _jsonl(run_dir / "scores.jsonl")
    predictions = _jsonl(run_dir / "predictions.jsonl")
    errors = _jsonl(run_dir / "errors.jsonl")
    expected_keys = {(call_id, item["id"]) for call_id in manifest["call_ids"] for item in dataset["checklist"]}
    for label, rows in (("results.csv", results), ("scores.jsonl", scores), ("predictions.jsonl", predictions)):
        keys = {(row.get("call_id"), row.get("question_id")) for row in rows}
        if keys != expected_keys or len(rows) != len(expected_keys):
            raise PublicationError(f"{label} does not contain exactly one row per expected call/question")
    return ValidatedPublication(
        evaluation_root=evaluation_root,
        run_dir=run_dir,
        dataset=dataset,
        metadata=metadata,
        summary=_json(run_dir / "summary.json"),
        results=results,
        scores=scores,
        predictions=predictions,
        errors=errors,
        artifact_fingerprint=directory_fingerprint(run_dir),
    )


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_rows(publication: ValidatedPublication) -> list[dict[str, Any]]:
    definitions = _json(publication.evaluation_root / "metric_definitions.json")
    rows: list[dict[str, Any]] = []
    scopes = [("overall", "", publication.summary.get("overall", {}))]
    scopes += [("call", key, value) for key, value in publication.summary.get("by_call_id", {}).items()]
    scopes += [("question", key, value) for key, value in publication.summary.get("by_question_id", {}).items()]
    for scope_type, scope_id, node in scopes:
        for definition in definitions.get("metrics", []):
            payload = nested(node, definition["summary_path"])
            if payload is None:
                continue
            rows.append({
                "run_id": publication.metadata["run_id"],
                "scope_type": scope_type,
                "scope_id": scope_id,
                "evaluation_family": definition["applies_to"],
                "metric_id": definition["id"],
                "numerator": _number(payload.get("numerator")) if isinstance(payload, dict) else None,
                "denominator": _number(payload.get("denominator")) if isinstance(payload, dict) else None,
                "value": _number(payload.get("value")) if isinstance(payload, dict) else None,
                "percentage": _number(payload.get("percentage")) if isinstance(payload, dict) else None,
                "payload": payload,
            })
    return rows


class SupabasePublisher:
    def __init__(self, client: Any, progress: Any = None):
        self.client = client
        self.bucket = client.storage.from_(BUCKET)
        self._progress = progress

    def _say(self, message: str) -> None:
        callback = getattr(self, "_progress", None)
        if callback:
            callback(message)

    def published_run_ids(self) -> set[str]:
        """Run IDs already published, for the upload picker. Best effort."""
        rows = self.client.table("evaluation_runs").select("run_id").eq("publication_state", "published").execute().data
        return {row["run_id"] for row in rows}

    @classmethod
    def from_environment(cls, progress: Any = None) -> "SupabasePublisher":
        from supabase import create_client

        required = {
            "SUPABASE_URL": os.getenv("SUPABASE_URL"),
            "SUPABASE_PUBLISHABLE_KEY": os.getenv("SUPABASE_PUBLISHABLE_KEY"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise PublicationError(f"Missing environment variables: {', '.join(missing)}")
        # Publishing needs no login: migration 20261004000000_allow_keyholder_publication
        # lets the publishable key insert and finalize records, never edit or delete them.
        try:
            client = create_client(required["SUPABASE_URL"], required["SUPABASE_PUBLISHABLE_KEY"])
        except Exception as error:
            raise PublicationError(f"Supabase client failed ({type(error).__name__}).") from error
        return cls(client, progress)

    def _select_one(self, table: str, column: str, value: str) -> dict[str, Any] | None:
        response = self.client.table(table).select("*").eq(column, value).limit(1).execute()
        return response.data[0] if response.data else None

    def _ensure_object(self, object_path: str, content: bytes) -> None:
        content_type = mimetypes.guess_type(object_path)[0] or "application/octet-stream"
        try:
            self.bucket.upload(object_path, content, file_options={"content-type": content_type, "upsert": "false"})
            return
        except Exception as upload_error:
            try:
                existing = self.bucket.download(object_path)
            except Exception:
                existing = None
            if existing is None:
                raise PublicationError(
                    f"Upload failed for {object_path} ({type(upload_error).__name__}): {upload_error}"
                ) from upload_error
            if existing != content:
                raise PublicationError(f"Storage integrity collision: {object_path}")

    def _ensure_objects(self, prefix: str, root: Path) -> None:
        files = sorted(item for item in root.rglob("*") if item.is_file())
        with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
            futures = [
                pool.submit(self._ensure_object, f"{prefix}/{path.relative_to(root).as_posix()}", path.read_bytes())
                for path in files
            ]
            for future in futures:
                future.result()

    def _existing_rows(self, table: str, scope: dict[str, str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            query = self.client.table(table).select("*")
            for column, value in scope.items():
                query = query.eq(column, value)
            page = query.range(offset, offset + 999).execute().data
            rows.extend(page)
            if len(page) < 1000:
                return rows
            offset += 1000

    def _ensure_rows(
        self,
        table: str,
        key_columns: tuple[str, ...],
        rows: list[dict[str, Any]],
        scope: dict[str, str],
    ) -> None:
        """Insert the rows that are missing and verify the ones that already exist.

        One read per table (not per row), then chunked inserts: a run is hundreds of rows.
        """
        existing = {tuple(row[key] for key in key_columns): row for row in self._existing_rows(table, scope)}
        missing = []
        for row in rows:
            key = tuple(row[column] for column in key_columns)
            current = existing.get(key)
            if current is None:
                missing.append(row)
                continue
            for column, value in row.items():
                if current.get(column) != value:
                    raise PublicationError(f"Database integrity collision in {table}: {key}")
        for start in range(0, len(missing), INSERT_CHUNK):
            self.client.table(table).insert(missing[start:start + INSERT_CHUNK]).execute()

    def _publish_dataset(self, publication: ValidatedPublication) -> None:
        dataset = publication.dataset
        manifest = dataset["manifest"]
        dataset_hash = manifest["dataset_sha256"]
        prefix = f"datasets/{dataset_hash}"
        dataset_row = {
            "dataset_sha256": dataset_hash,
            "dataset_name": dataset["root"].name,
            "checklist_sha256": manifest["files"]["checklist.json"],
            "manifest_sha256": dataset["manifest_sha256"],
            "dataset_format": manifest["format"],
            "call_ids": manifest["call_ids"],
            "question_count": len(dataset["checklist"]),
            "storage_prefix": prefix,
            "manifest": manifest,
            "publication_state": "publishing",
        }
        existing = self._select_one("evaluation_datasets", "dataset_sha256", dataset_hash)
        if existing:
            for key in ("checklist_sha256", "manifest_sha256", "dataset_format", "call_ids", "question_count", "storage_prefix"):
                if existing.get(key) != dataset_row[key]:
                    raise PublicationError(f"Dataset integrity collision for {dataset_hash}")
            if existing.get("publication_state") == "published":
                # A published snapshot is immutable and identified by its hash: nothing to redo.
                self._say("Dataset snapshot already published")
                return
        self._say(f"Uploading dataset files ({sum(1 for item in dataset['root'].rglob('*') if item.is_file())})")
        self._ensure_objects(prefix, dataset["root"])
        self._say("Writing dataset records")
        if not existing:
            self.client.table("evaluation_datasets").insert(dataset_row).execute()
        call_rows = []
        for call_id in manifest["call_ids"]:
            documents = call_documents(dataset, call_id)
            call_rows.append({"dataset_sha256": dataset_hash, "call_id": call_id, **documents})
        self._ensure_rows("evaluation_dataset_calls", ("dataset_sha256", "call_id"), call_rows, {"dataset_sha256": dataset_hash})
        checklist_rows = [{
            "dataset_sha256": dataset_hash,
            "question_id": item["id"],
            "question": item["question"],
            "evaluation_family": item["evaluation_family"],
            "question_type": item["type"],
            "requirement_type": item.get("requirement_type"),
            "field": item.get("field"),
            "definition": item,
        } for item in dataset["checklist"]]
        self._ensure_rows("evaluation_checklist_items", ("dataset_sha256", "question_id"), checklist_rows, {"dataset_sha256": dataset_hash})
        if not existing or existing.get("publication_state") != "published":
            self.client.table("evaluation_datasets").update({"publication_state": "published"}).eq("dataset_sha256", dataset_hash).execute()

    def publish(self, publication: ValidatedPublication) -> str:
        run_id = publication.metadata["run_id"]
        existing = self._select_one("evaluation_runs", "run_id", run_id)
        if existing and existing.get("artifact_fingerprint_sha256") != publication.artifact_fingerprint:
            raise PublicationError(f"Run-ID integrity collision for {run_id}")
        if existing and existing.get("publication_state") == "published":
            return "already_published"
        self._say("Checking the shared dataset snapshot")
        self._publish_dataset(publication)
        prefix = f"runs/{run_id}"
        self._say(f"Uploading run files ({sum(1 for item in publication.run_dir.rglob('*') if item.is_file())})")
        self._ensure_objects(prefix, publication.run_dir)
        self._say("Writing run record")
        metadata = publication.metadata
        snapshot = metadata["dataset_snapshot"]
        run_row = {
            "run_id": run_id,
            "dataset_sha256": snapshot["dataset_sha256"],
            "artifact_fingerprint_sha256": publication.artifact_fingerprint,
            "started_at": metadata.get("started_at"),
            "completed_at": metadata.get("completed_at"),
            "execution_status": metadata["execution_status"],
            "run_kind": metadata["run_kind"],
            "artifact_schema_version": metadata["artifact_schema_version"],
            "scorer_version": metadata["scorer_version"],
            "metric_definitions_version": metadata["metric_definitions_version"],
            "metric_definitions_sha256": metadata["metric_definitions_sha256"],
            "checklist_sha256": snapshot["checklist_sha256"],
            "graph_commit": metadata.get("graph_commit"),
            "git_commit_author": metadata.get("git_commit_author"),
            "git_commit_timestamp": metadata.get("git_commit_timestamp"),
            "graph_worktree_dirty": metadata.get("graph_worktree_dirty"),
            "total_latency_ms": metadata.get("total_latency_ms"),
            "selected_call_ids": metadata["execution_configuration"]["selected_call_ids"],
            "storage_prefix": prefix,
            "metadata": metadata,
            "summary": publication.summary,
            "publication_state": "publishing",
        }
        if not existing:
            self.client.table("evaluation_runs").insert(run_row).execute()
        scores = {(row["call_id"], row["question_id"]): row for row in publication.scores}
        predictions = {(row["call_id"], row["question_id"]): row for row in publication.predictions}
        result_rows = []
        for row in publication.results:
            key = (row["call_id"], row["question_id"])
            result_rows.append({
                "run_id": run_id,
                "call_id": row["call_id"],
                "question_id": row["question_id"],
                "evaluation_family": row["evaluation_family"],
                "execution_status": row["execution_status"],
                "failure_category": row.get("failure_category") or None,
                "gold_verdict": row.get("gold_verdict") or None,
                "predicted_verdict": row.get("predicted_verdict") or None,
                "compliance_outcome_status": row.get("compliance_outcome_status") or None,
                "confusion_category": row.get("confusion_category") or None,
                "breach_outcome": row.get("breach_outcome") or None,
                "expected_value": row.get("expected_value") or None,
                "gold_value_status": row.get("gold_value_status") or None,
                "extraction_evidence_status": row.get("extraction_evidence_status") or None,
                "evidence_precision": _number(row.get("evidence_precision")),
                "evidence_recall": _number(row.get("evidence_recall")),
                "evidence_exact_set_match": _number(row.get("evidence_exact_set_match")),
                "result_row": row,
                "score": scores[key],
                "prediction": predictions[key],
            })
        self._say(f"Writing {len(result_rows)} results")
        self._ensure_rows("evaluation_run_results", ("run_id", "call_id", "question_id"), result_rows, {"run_id": run_id})
        metrics = metric_rows(publication)
        self._say(f"Writing {len(metrics)} metric values")
        self._ensure_rows("evaluation_run_metrics", ("run_id", "scope_type", "scope_id", "metric_id"), metrics, {"run_id": run_id})
        error_rows = [{
            "run_id": run_id,
            "error_index": index,
            "call_id": error.get("call_id"),
            "question_id": error.get("question_id"),
            "category": error.get("category"),
            "message": error.get("message"),
            "payload": error,
        } for index, error in enumerate(publication.errors)]
        self._ensure_rows("evaluation_run_errors", ("run_id", "error_index"), error_rows, {"run_id": run_id})
        self._say("Marking the run as published")
        self.client.table("evaluation_runs").update({"publication_state": "published"}).eq("run_id", run_id).execute()
        return "published"
