"""Filesystem implementation of the evaluation repository."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from fyp_evaluation_loader import call_documents, load_dataset

from .domain import DatasetRecord, ResultRecord, RunRecord
from .repository import EvaluationRepository, RepositoryError


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise RepositoryError(f"Cannot read {path}: {error}") from error


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise RepositoryError(f"Cannot read {path}: {error}") from error


class LocalEvaluationRepository(EvaluationRepository):
    source_name = "Local"

    def __init__(self, evaluation_root: str | Path):
        self.root = Path(evaluation_root).resolve()
        self.datasets_root = self.root / "datasets"
        self.runs_root = self.root / "runs"
        self._datasets: dict[str, tuple[DatasetRecord, dict[str, Any]]] | None = None

    def _load_datasets(self) -> dict[str, tuple[DatasetRecord, dict[str, Any]]]:
        if self._datasets is not None:
            return self._datasets
        loaded: dict[str, tuple[DatasetRecord, dict[str, Any]]] = {}
        if not self.datasets_root.is_dir():
            return loaded
        for path in sorted(item for item in self.datasets_root.iterdir() if item.is_dir()):
            try:
                dataset = load_dataset(path)
            except Exception as error:
                raise RepositoryError(f"Invalid local dataset {path.name}: {error}") from error
            manifest = dataset["manifest"]
            dataset_hash = manifest.get("dataset_sha256")
            if not dataset_hash:
                raise RepositoryError(f"Dataset {path.name} has no dataset_sha256")
            record = DatasetRecord(
                dataset_id=dataset_hash,
                name=path.name,
                dataset_sha256=dataset_hash,
                checklist_sha256=manifest.get("files", {}).get("checklist.json"),
                manifest_sha256=dataset["manifest_sha256"],
                dataset_format=manifest["format"],
                call_ids=tuple(manifest["call_ids"]),
                question_count=len(dataset["checklist"]),
                source=self.source_name,
                manifest=manifest,
            )
            loaded[dataset_hash] = (record, dataset)
        self._datasets = loaded
        return loaded

    def list_datasets(self) -> list[DatasetRecord]:
        return sorted((pair[0] for pair in self._load_datasets().values()), key=lambda item: item.name.lower())

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        try:
            return self._load_datasets()[dataset_id][0]
        except KeyError:
            raise RepositoryError(f"Dataset not found: {dataset_id}") from None

    def _dataset_documents(self, dataset_id: str) -> dict[str, Any]:
        try:
            return self._load_datasets()[dataset_id][1]
        except KeyError:
            raise RepositoryError(f"Dataset not found: {dataset_id}") from None

    def get_checklist(self, dataset_id: str) -> list[dict[str, Any]]:
        return list(self._dataset_documents(dataset_id)["checklist"])

    def _run_from_dir(self, run_dir: Path) -> RunRecord:
        metadata = _read_json(run_dir / "metadata.json")
        summary = _read_json(run_dir / "summary.json")
        dataset_id = metadata.get("dataset_snapshot", {}).get("dataset_sha256")
        dataset = self.get_dataset(dataset_id)
        selected = set(metadata.get("execution_configuration", {}).get("selected_call_ids") or
                       metadata.get("dataset_snapshot", {}).get("call_ids") or [])
        complete = selected == set(dataset.call_ids)
        return RunRecord(
            run_id=str(metadata.get("run_id") or run_dir.name),
            dataset_id=dataset_id,
            source=self.source_name,
            metadata=metadata,
            summary=summary,
            is_complete=complete,
        )

    def list_runs(self, dataset_id: str) -> list[RunRecord]:
        self.get_dataset(dataset_id)
        runs: list[RunRecord] = []
        if not self.runs_root.is_dir():
            return runs
        for path in self.runs_root.iterdir():
            if not path.is_dir() or not (path / "metadata.json").is_file() or not (path / "summary.json").is_file():
                continue
            try:
                run = self._run_from_dir(path)
            except RepositoryError:
                continue
            if run.dataset_id == dataset_id:
                runs.append(run)
        return sorted(runs, key=lambda item: item.evaluated_at or "", reverse=True)

    def list_unlinked_runs(self) -> list[dict[str, Any]]:
        unlinked: list[dict[str, Any]] = []
        if not self.runs_root.is_dir():
            return unlinked
        known = set(self._load_datasets())
        for path in sorted(self.runs_root.iterdir()):
            # A folder without both files is a run still being written, not a broken one.
            if not path.is_dir() or not (path / "metadata.json").is_file() or not (path / "summary.json").is_file():
                continue
            entry = {"run_id": path.name, "evaluated_at": None, "dataset_sha256": None}
            try:
                metadata = _read_json(path / "metadata.json")
                entry["evaluated_at"] = metadata.get("started_at")
                entry["dataset_sha256"] = (metadata.get("dataset_snapshot") or {}).get("dataset_sha256")
                if entry["dataset_sha256"] in known:
                    self._run_from_dir(path)  # listed normally unless it is unreadable
                    continue
                unlinked.append({**entry, "reason": "dataset_missing"})
            except RepositoryError:
                unlinked.append({**entry, "reason": "unreadable"})
        return unlinked

    def get_run(self, run_id: str) -> RunRecord:
        run_dir = self.runs_root / run_id
        if not run_dir.is_dir():
            raise RepositoryError(f"Run not found: {run_id}")
        return self._run_from_dir(run_dir)

    def _result_rows(self, run_id: str) -> list[dict[str, Any]]:
        path = self.runs_root / run_id / "results.csv"
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                return list(csv.DictReader(handle))
        except OSError as error:
            raise RepositoryError(f"Cannot read {path}: {error}") from error

    def list_results(
        self,
        run_id: str,
        *,
        call_id: str | None = None,
        question_id: str | None = None,
        family: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._result_rows(run_id)
        return [
            row for row in rows
            if (call_id is None or row.get("call_id") == call_id)
            and (question_id is None or row.get("question_id") == question_id)
            and (family is None or row.get("evaluation_family") == family)
        ]

    def list_scores(self, run_id: str) -> list[dict[str, Any]]:
        self.get_run(run_id)
        return _read_jsonl(self.runs_root / run_id / "scores.jsonl")

    def get_call_transcript(self, run_id: str, call_id: str) -> tuple[dict[str, Any], dict[str, str]]:
        run = self.get_run(run_id)
        documents = call_documents(self._dataset_documents(run.dataset_id), call_id)
        return documents["transcript"], documents["speaker_roles"]

    def get_result(self, run_id: str, call_id: str, question_id: str) -> ResultRecord:
        run = self.get_run(run_id)
        matches = self.list_results(run_id, call_id=call_id, question_id=question_id)
        if len(matches) != 1:
            raise RepositoryError(f"Expected one result for {run_id}/{call_id}/{question_id}")
        run_dir = self.runs_root / run_id
        scores = {(item["call_id"], item["question_id"]): item for item in _read_jsonl(run_dir / "scores.jsonl")}
        predictions = {(item["call_id"], item["question_id"]): item for item in _read_jsonl(run_dir / "predictions.jsonl")}
        checklist = {item["id"]: item for item in self.get_checklist(run.dataset_id)}
        documents = call_documents(self._dataset_documents(run.dataset_id), call_id)
        ground_truth = {item["question_id"]: item for item in documents["ground_truth"]["checks"]}
        key = (call_id, question_id)
        return ResultRecord(
            run_id=run_id,
            call_id=call_id,
            question_id=question_id,
            row=matches[0],
            score=scores.get(key, {}),
            prediction=predictions.get(key, {}),
            question=checklist.get(question_id, {"id": question_id, "question": question_id}),
            ground_truth=ground_truth.get(question_id, {}),
            transcript=documents["transcript"],
            speaker_roles=documents["speaker_roles"],
        )

    def get_errors(self, run_id: str) -> list[dict[str, Any]]:
        path = self.runs_root / run_id / "errors.jsonl"
        return _read_jsonl(path) if path.is_file() else []

    def get_metric_definitions(self) -> dict[str, Any]:
        return _read_json(self.root / "metric_definitions.json")

    def run_exists(self, run_id: str) -> bool:
        return (self.runs_root / run_id / "metadata.json").is_file()
