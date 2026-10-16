"""Repository contract shared by local files and Supabase."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .domain import DatasetRecord, ResultRecord, RunRecord


class RepositoryError(RuntimeError):
    """The selected evaluation source is unavailable or inconsistent."""


class EvaluationRepository(ABC):
    source_name: str

    @abstractmethod
    def list_datasets(self) -> list[DatasetRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        raise NotImplementedError

    @abstractmethod
    def get_checklist(self, dataset_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_runs(self, dataset_id: str) -> list[RunRecord]:
        raise NotImplementedError

    def list_unlinked_runs(self) -> list[dict[str, Any]]:
        """Runs that exist but cannot be listed under any dataset, with the reason.

        Only the local source can have these (a run made on a dataset version that is no longer
        on disk); a published run always has its dataset.
        """
        return []

    @abstractmethod
    def get_run(self, run_id: str) -> RunRecord:
        raise NotImplementedError

    @abstractmethod
    def list_results(
        self,
        run_id: str,
        *,
        call_id: str | None = None,
        question_id: str | None = None,
        family: str | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def list_scores(self, run_id: str) -> list[dict[str, Any]]:
        """Typed scores.jsonl records of one run, one per call-question."""
        raise NotImplementedError

    @abstractmethod
    def get_call_transcript(self, run_id: str, call_id: str) -> tuple[dict[str, Any], dict[str, str]]:
        """The transcript and speaker roles of one call of a run's dataset."""
        raise NotImplementedError

    @abstractmethod
    def get_result(self, run_id: str, call_id: str, question_id: str) -> ResultRecord:
        raise NotImplementedError

    @abstractmethod
    def get_errors(self, run_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_metric_definitions(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def run_exists(self, run_id: str) -> bool:
        raise NotImplementedError
