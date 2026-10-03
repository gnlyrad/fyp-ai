"""Supabase implementation of the read-only evaluation repository."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from supabase import Client, create_client

from .domain import DatasetRecord, ResultRecord, RunRecord
from .repository import EvaluationRepository, RepositoryError


class SupabaseEvaluationRepository(EvaluationRepository):
    source_name = "Shared"

    def __init__(self, client: Client, metric_definitions_path: str | Path):
        self.client = client
        self.metric_definitions_path = Path(metric_definitions_path)

    @staticmethod
    def _all_rows(query: Any, page_size: int = 1000) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = query.range(offset, offset + page_size - 1).execute().data
            rows.extend(page)
            if len(page) < page_size:
                return rows
            offset += page_size

    @classmethod
    def from_environment(cls) -> "SupabaseEvaluationRepository":
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_PUBLISHABLE_KEY")
        if not all((url, key)):
            raise RepositoryError(
                "Shared source requires SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY."
            )
        definitions = Path(__file__).resolve().parents[1] / "metric_definitions.json"
        try:
            client = create_client(url, key)
        except Exception as error:
            raise RepositoryError(f"Supabase dashboard client failed ({type(error).__name__}).") from error
        return cls(client, definitions)

    @staticmethod
    def _dataset(row: dict[str, Any]) -> DatasetRecord:
        return DatasetRecord(
            dataset_id=row["dataset_sha256"],
            name=row["dataset_name"],
            dataset_sha256=row["dataset_sha256"],
            checklist_sha256=row.get("checklist_sha256"),
            manifest_sha256=row.get("manifest_sha256"),
            dataset_format=row["dataset_format"],
            call_ids=tuple(row.get("call_ids") or []),
            question_count=int(row["question_count"]),
            source="Shared",
            manifest=row.get("manifest") or {},
        )

    @staticmethod
    def _run(row: dict[str, Any]) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            dataset_id=row["dataset_sha256"],
            source="Shared",
            metadata=row.get("metadata") or {},
            summary=row.get("summary") or {},
            is_complete=True,
        )

    def list_datasets(self) -> list[DatasetRecord]:
        query = self.client.table("evaluation_datasets").select("*").eq("publication_state", "published").order("dataset_name")
        return [self._dataset(row) for row in self._all_rows(query)]

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        response = self.client.table("evaluation_datasets").select("*").eq("dataset_sha256", dataset_id).eq("publication_state", "published").limit(1).execute()
        if not response.data:
            raise RepositoryError(f"Shared dataset not found: {dataset_id}")
        return self._dataset(response.data[0])

    def get_checklist(self, dataset_id: str) -> list[dict[str, Any]]:
        query = self.client.table("evaluation_checklist_items").select("definition").eq("dataset_sha256", dataset_id).order("question_id")
        return [row["definition"] for row in self._all_rows(query)]

    def list_runs(self, dataset_id: str) -> list[RunRecord]:
        query = self.client.table("evaluation_runs").select("*").eq("dataset_sha256", dataset_id).eq("publication_state", "published").order("started_at", desc=True)
        return [self._run(row) for row in self._all_rows(query)]

    def get_run(self, run_id: str) -> RunRecord:
        response = self.client.table("evaluation_runs").select("*").eq("run_id", run_id).eq("publication_state", "published").limit(1).execute()
        if not response.data:
            raise RepositoryError(f"Shared run not found: {run_id}")
        return self._run(response.data[0])

    def list_results(
        self,
        run_id: str,
        *,
        call_id: str | None = None,
        question_id: str | None = None,
        family: str | None = None,
    ) -> list[dict[str, Any]]:
        query = self.client.table("evaluation_run_results").select("result_row").eq("run_id", run_id)
        if call_id:
            query = query.eq("call_id", call_id)
        if question_id:
            query = query.eq("question_id", question_id)
        if family:
            query = query.eq("evaluation_family", family)
        query = query.order("call_id").order("question_id")
        return [row["result_row"] for row in self._all_rows(query)]

    def list_scores(self, run_id: str) -> list[dict[str, Any]]:
        query = self.client.table("evaluation_run_results").select("score").eq("run_id", run_id)
        query = query.order("call_id").order("question_id")
        return [row["score"] for row in self._all_rows(query)]

    def get_result(self, run_id: str, call_id: str, question_id: str) -> ResultRecord:
        run = self.get_run(run_id)
        result_response = self.client.table("evaluation_run_results").select("*").eq("run_id", run_id).eq("call_id", call_id).eq("question_id", question_id).limit(1).execute()
        if not result_response.data:
            raise RepositoryError(f"Shared result not found: {run_id}/{call_id}/{question_id}")
        call_response = self.client.table("evaluation_dataset_calls").select("transcript,speaker_roles,ground_truth").eq("dataset_sha256", run.dataset_id).eq("call_id", call_id).limit(1).execute()
        questions = {item["id"]: item for item in self.get_checklist(run.dataset_id)}
        if not call_response.data:
            raise RepositoryError(f"Shared call not found: {call_id}")
        row = result_response.data[0]
        call = call_response.data[0]
        ground_truth = {item["question_id"]: item for item in call["ground_truth"]["checks"]}
        return ResultRecord(
            run_id=run_id,
            call_id=call_id,
            question_id=question_id,
            row=row["result_row"],
            score=row["score"],
            prediction=row["prediction"],
            question=questions[question_id],
            ground_truth=ground_truth[question_id],
            transcript=call["transcript"],
            speaker_roles=call["speaker_roles"],
        )

    def get_errors(self, run_id: str) -> list[dict[str, Any]]:
        query = self.client.table("evaluation_run_errors").select("payload").eq("run_id", run_id).order("error_index")
        return [row["payload"] for row in self._all_rows(query)]

    def get_metric_definitions(self) -> dict[str, Any]:
        try:
            return json.loads(self.metric_definitions_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            raise RepositoryError(f"Cannot read metric definitions: {error}") from error

    def run_exists(self, run_id: str) -> bool:
        response = self.client.table("evaluation_runs").select("run_id").eq("run_id", run_id).eq("publication_state", "published").limit(1).execute()
        return bool(response.data)
