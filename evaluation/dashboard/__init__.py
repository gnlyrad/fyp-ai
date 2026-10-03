"""Read-only dashboard over immutable evaluation artifacts."""

from .domain import DatasetRecord, ResultRecord, RunRecord
from .local_repository import LocalEvaluationRepository
from .repository import EvaluationRepository, RepositoryError

__all__ = [
    "DatasetRecord",
    "EvaluationRepository",
    "LocalEvaluationRepository",
    "RepositoryError",
    "ResultRecord",
    "RunRecord",
]
