"""Storage-independent records consumed by the dashboard views."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    name: str
    dataset_sha256: str
    checklist_sha256: str | None
    manifest_sha256: str | None
    dataset_format: str
    call_ids: tuple[str, ...]
    question_count: int
    source: str
    manifest: dict[str, Any] = field(repr=False)

    @property
    def call_count(self) -> int:
        return len(self.call_ids)


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    dataset_id: str
    source: str
    metadata: dict[str, Any] = field(repr=False)
    summary: dict[str, Any] = field(repr=False)
    is_complete: bool = False

    @property
    def status(self) -> str:
        return str(self.metadata.get("execution_status") or "unknown")

    @property
    def evaluated_at(self) -> str | None:
        return self.metadata.get("started_at")


@dataclass(frozen=True)
class ResultRecord:
    run_id: str
    call_id: str
    question_id: str
    row: dict[str, Any]
    score: dict[str, Any]
    prediction: dict[str, Any]
    question: dict[str, Any]
    ground_truth: dict[str, Any]
    transcript: dict[str, Any]
    speaker_roles: dict[str, str]

    @property
    def evaluation_family(self) -> str:
        return str(self.row.get("evaluation_family") or self.score.get("evaluation_family") or "")
