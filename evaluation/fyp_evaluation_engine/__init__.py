"""Deterministic scoring and persistence for the real ClearCall graph."""

from .adapter import normalize_call_output
from .scoring import aggregate_scores, score_prediction

SCORER_VERSION = "fyp-nlp-evaluator-v2.0"
METRIC_DEFINITIONS_VERSION = "2.0.0"
ARTIFACT_SCHEMA_VERSION = "2.0.0"

__all__ = [
    "ARTIFACT_SCHEMA_VERSION", "METRIC_DEFINITIONS_VERSION", "SCORER_VERSION", "aggregate_scores",
    "normalize_call_output", "score_prediction",
]
