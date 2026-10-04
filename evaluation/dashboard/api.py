"""Dashboard API over EvaluationRepository, for both the Local and Shared sources.

The React app in evaluation/web talks only to this API; it never reads run
files or queries Supabase itself. When evaluation/web/dist exists (after
`npm run build`), the same server also serves the built app.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .domain import DatasetRecord
from .local_repository import LocalEvaluationRepository
from .presenters import (
    compare_runs,
    drills,
    evidence_parts,
    evidence_explanation,
    problems,
    result_outcome,
    result_row,
    run_status,
    run_summary,
    transcript_turns,
)
from .repository import EvaluationRepository, RepositoryError
from .view_models import comparison_incompatibilities, nested

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
WEB_DIST = EVALUATION_ROOT / "web" / "dist"
SOURCES = ("shared", "local")


def _default_shared() -> EvaluationRepository:
    from .supabase_repository import SupabaseEvaluationRepository  # noqa: PLC0415
    return SupabaseEvaluationRepository.from_environment()


class LiveJson(Mapping):
    """A JSON file that is re-read whenever it changes on disk.

    The wording files are edited while the dashboard is running; this way a wording change shows
    up on the next page load without restarting the server. Only data files reload: code changes
    still need a restart.
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._stamp: tuple[int, int] | None = None
        self._value: dict[str, Any] = {}

    def _current(self) -> dict[str, Any]:
        stat = self._path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp != self._stamp:
            self._value = json.loads(self._path.read_text(encoding="utf-8-sig"))
            self._stamp = stamp
        return self._value

    def __getitem__(self, key: str) -> Any:
        return self._current()[key]

    def __iter__(self):
        return iter(self._current())

    def __len__(self) -> int:
        return len(self._current())


class SourceUnavailable(RuntimeError):
    pass


def dataset_label(dataset: DatasetRecord) -> str:
    return dataset.name.replace("_", " ").title()


def create_app(
    *,
    evaluation_root: str | Path = EVALUATION_ROOT,
    shared_factory: Callable[[], EvaluationRepository] = _default_shared,
    web_dist: str | Path | None = WEB_DIST,
    default_source: str | None = None,
) -> FastAPI:
    evaluation_root = Path(evaluation_root)
    definitions = LiveJson(evaluation_root / "metric_definitions.json")
    presentation = LiveJson(evaluation_root / "metric_presentation.json")
    shared_cache: dict[str, EvaluationRepository] = {}
    chosen_default = (default_source or os.getenv("EVALUATION_DEFAULT_SOURCE", "shared")).lower()
    if chosen_default not in SOURCES:
        chosen_default = "shared"

    app = FastAPI(title="FYP evaluation dashboard API")

    def repository(source: str) -> EvaluationRepository:
        if source == "local":
            # Fresh per request so runs and datasets added on disk appear without a restart.
            return LocalEvaluationRepository(evaluation_root)
        if source != "shared":
            raise HTTPException(404, f"Unknown source: {source}")
        if "shared" not in shared_cache:
            try:
                shared_cache["shared"] = shared_factory()
            except Exception as error:  # missing env vars or client failure
                raise SourceUnavailable(str(error) if isinstance(error, RepositoryError)
                                        else f"Shared source failed to start ({type(error).__name__}).") from error
        return shared_cache["shared"]

    @app.exception_handler(RepositoryError)
    async def _repository_error(_request, error: RepositoryError):
        return JSONResponse({"detail": str(error)}, status_code=404)

    @app.exception_handler(SourceUnavailable)
    async def _unavailable(_request, error: SourceUnavailable):
        return JSONResponse({"detail": str(error)}, status_code=503)

    def guarded(source: str, action: Callable[[EvaluationRepository], Any]) -> Any:
        repo = repository(source)
        try:
            return action(repo)
        except (RepositoryError, HTTPException):
            raise
        except Exception as error:
            if source != "shared":
                raise
            raise SourceUnavailable(f"Shared evaluation source is unavailable ({type(error).__name__}).") from error

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "default_source": chosen_default,
            "sources": list(SOURCES),
            "definitions": dict(definitions),
            "presentation": dict(presentation),
            "drills": drills(presentation),
        }

    def dataset_view(repo: EvaluationRepository, dataset: DatasetRecord) -> dict[str, Any]:
        families = Counter(item.get("evaluation_family") for item in repo.get_checklist(dataset.dataset_id))
        return {
            "dataset_id": dataset.dataset_id,
            "name": dataset.name,
            "label": dataset_label(dataset),
            "call_ids": list(dataset.call_ids),
            "call_count": dataset.call_count,
            "question_count": dataset.question_count,
            "compliance_questions": families.get("compliance", 0),
            "extraction_questions": families.get("extraction", 0),
            "dataset_sha256": dataset.dataset_sha256,
            "checklist_sha256": dataset.checklist_sha256,
            "manifest_sha256": dataset.manifest_sha256,
            "format": dataset.dataset_format,
        }

    @app.get("/api/sources/{source}/datasets")
    def list_datasets(source: str) -> list[dict[str, Any]]:
        def action(repo: EvaluationRepository) -> list[dict[str, Any]]:
            items = []
            for dataset in repo.list_datasets():
                runs = repo.list_runs(dataset.dataset_id)
                latest = runs[0] if runs else None
                items.append({
                    **dataset_view(repo, dataset),
                    "run_count": len(runs),
                    "latest_run_at": latest.evaluated_at if latest else None,
                })
            return items
        return guarded(source, action)

    @app.get("/api/sources/{source}/datasets/{dataset_id}")
    def get_dataset(source: str, dataset_id: str) -> dict[str, Any]:
        def action(repo: EvaluationRepository) -> dict[str, Any]:
            dataset = repo.get_dataset(dataset_id)
            runs = repo.list_runs(dataset_id)
            return {
                "dataset": dataset_view(repo, dataset),
                "runs": [run_summary(run, definitions, presentation) for run in runs],
            }
        return guarded(source, action)

    @app.get("/api/sources/{source}/runs/{run_id}")
    def get_run(source: str, run_id: str) -> dict[str, Any]:
        def action(repo: EvaluationRepository) -> dict[str, Any]:
            run = repo.get_run(run_id)
            dataset = repo.get_dataset(run.dataset_id)
            checklist = repo.get_checklist(run.dataset_id)
            questions = {item["id"]: item for item in checklist}
            errors = repo.get_errors(run_id)
            rows = []
            for score in repo.list_scores(run_id):
                row = result_row(score, questions.get(score["question_id"]))
                rows.append({**row, "outcome": result_outcome(row, presentation)})
            return {
                "run": {
                    **run_summary(run, definitions, presentation),
                    "status": run_status(run, errors, presentation),
                    "completed_at": run.metadata.get("completed_at"),
                    "source": run.source,
                    "metadata": run.metadata,
                },
                "dataset": {"dataset_id": dataset.dataset_id, "label": dataset_label(dataset), "name": dataset.name},
                "checklist": checklist,
                "summary": run.summary,
                "rows": rows,
                "problems": problems(run.summary.get("overall", {}), errors, presentation),
            }
        return guarded(source, action)

    @app.get("/api/sources/{source}/runs/{run_id}/results/{call_id}/{question_id}")
    def get_result(source: str, run_id: str, call_id: str, question_id: str) -> dict[str, Any]:
        def action(repo: EvaluationRepository) -> dict[str, Any]:
            result = repo.get_result(run_id, call_id, question_id)
            row = result_row(result.score, result.question)
            turns = transcript_turns(result.transcript, result.speaker_roles, row)
            finding = nested(result.prediction, "graph_metadata.finding", {}) or {}
            return {
                "row": {**row, "outcome": result_outcome(row, presentation)},
                "question": result.question,
                "turns": turns,
                "explanation": evidence_explanation(row),
                "selected_quotes": [
                    span.get("text") for span in result.prediction.get("selected_evidence") or []
                    if isinstance(span, dict) and isinstance(span.get("text"), str)
                ],
                "reasoning": finding.get("reason") or finding.get("reasoning"),
                "failure_message": result.prediction.get("failure_message"),
                "prediction": result.prediction,
                "score": result.score,
                "ground_truth": result.ground_truth,
            }
        return guarded(source, action)

    @app.get("/api/sources/{source}/runs/{run_id}/diagnostics")
    def get_diagnostics(source: str, run_id: str) -> dict[str, Any]:
        def action(repo: EvaluationRepository) -> dict[str, Any]:
            repo.get_run(run_id)
            results = {(score["call_id"], score["question_id"]) for score in repo.list_scores(run_id)}
            labels = presentation["error_categories"]
            errors = []
            for index, error in enumerate(repo.get_errors(run_id)):
                category = error.get("category")
                entry = labels.get(category, {})
                errors.append({
                    "index": index,
                    "category": category,
                    "label": entry.get("label", category),
                    "description": entry.get("description"),
                    "call_id": error.get("call_id"),
                    "question_id": error.get("question_id"),
                    "message": error.get("message"),
                    "linkable": (error.get("call_id"), error.get("question_id")) in results,
                    "payload": error,
                })
            return {"errors": errors, "counts": dict(Counter(error["category"] for error in errors))}
        return guarded(source, action)

    @app.get("/api/sources/{source}/runs/{run_id}/evidence")
    def get_evidence(source: str, run_id: str) -> dict[str, Any]:
        """Per-check missed and extra evidence turns (with text), for the Diagnostics drill-down."""
        def action(repo: EvaluationRepository) -> dict[str, Any]:
            run = repo.get_run(run_id)
            questions = {item["id"]: item for item in repo.get_checklist(run.dataset_id)}
            rows = []
            for score in repo.list_scores(run_id):
                row = result_row(score, questions.get(score["question_id"]))
                rows.append({**row, "outcome": result_outcome(row, presentation)})
            transcripts = {call_id: repo.get_call_transcript(run_id, call_id) for call_id in sorted({row["call_id"] for row in rows})}
            return {"entries": evidence_parts(rows, transcripts)}
        return guarded(source, action)

    @app.get("/api/sources/{source}/compare")
    def compare(source: str, a: str, b: str) -> dict[str, Any]:
        def action(repo: EvaluationRepository) -> dict[str, Any]:
            first, second = repo.get_run(a), repo.get_run(b)
            reasons = comparison_incompatibilities(first, second)
            result = compare_runs(first, second, definitions, reasons)
            summaries = {run.run_id: run_summary(run, definitions, presentation) for run in (first, second)}
            return {
                "earlier": summaries[result["earlier_run_id"]],
                "later": summaries[result["later_run_id"]],
                **result,
            }
        return guarded(source, action)

    _mount_frontend(app, Path(web_dist) if web_dist else None)
    return app


def _mount_frontend(app: FastAPI, web_dist: Path | None) -> None:
    index = web_dist / "index.html" if web_dist else None
    if index is None or not index.is_file():
        @app.get("/", include_in_schema=False)
        def not_built() -> HTMLResponse:
            return HTMLResponse(
                "<h1>Dashboard frontend not built</h1><p>Run <code>npm install</code> and "
                "<code>npm run build</code> in <code>evaluation/web</code>, then restart this server. "
                "The API is available under <code>/api</code>.</p>"
            )
        return

    if (web_dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=web_dist / "assets"), name="assets")
    root = web_dist.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404, "Not found")
        candidate = (root / path).resolve()
        if path and candidate.is_file() and root in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(index)
