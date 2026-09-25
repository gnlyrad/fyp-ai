"""Immutable run-directory artifact serialization."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any


def to_jsonable(value: Any) -> Any:
    """Convert graph output into deterministic JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return to_jsonable(value.value)
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((to_jsonable(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    return {"__type__": f"{type(value).__module__}.{type(value).__qualname__}", "value": str(value)}


def _json_text(value: Any, *, compact: bool = False) -> str:
    kwargs = {"ensure_ascii": False, "sort_keys": True}
    if compact:
        kwargs["separators"] = (",", ":")
    else:
        kwargs["indent"] = 2
    return json.dumps(to_jsonable(value), **kwargs) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(_json_text(value), encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "".join(_json_text(record, compact=True) for record in records)
    path.write_text(content, encoding="utf-8", newline="\n")


CSV_COLUMNS = (
    # shared
    "call_id", "question_id", "question_type", "evaluation_family", "requirement_type", "field",
    "execution_status", "failure_category", "native_verdict",
    "gold_evidence_turn_ids", "predicted_evidence_turn_ids",
    "evidence_status", "evidence_precision", "evidence_recall", "evidence_exact_set_match",
    "evidence_references_valid", "invalid_evidence_reference_count", "unresolved_evidence_references",
    "call_latency_ms",
    # compliance only (blank for extraction rows)
    "gold_verdict", "predicted_verdict", "compliance_outcome_status", "confusion_category",
    "binary_correct", "breach_outcome",
    # extraction only (blank for compliance rows)
    "expected_value", "value_type", "gold_value_status", "evaluator_support",
    "extraction_evidence_status", "extraction_evidence_coverage",
)


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    """One (call_id, question_id) row; fields of the other family stay blank, never zero."""
    evidence = record["evidence"]
    compliance = record.get("compliance") or {}
    extraction = record.get("extraction") or {}
    return {
        "call_id": record["call_id"],
        "question_id": record["question_id"],
        "question_type": record["question_type"],
        "evaluation_family": record["evaluation_family"],
        "requirement_type": record.get("requirement_type"),
        "field": record.get("field"),
        "execution_status": record["execution_status"],
        "failure_category": record.get("failure_category"),
        "native_verdict": record.get("native_verdict"),
        "gold_evidence_turn_ids": json.dumps(record["gold_evidence_turn_ids"], ensure_ascii=False),
        "predicted_evidence_turn_ids": json.dumps(record["predicted_evidence_turn_ids"], ensure_ascii=False),
        "evidence_status": evidence["status"],
        "evidence_precision": evidence["precision"]["value"],
        "evidence_recall": evidence["recall"]["value"],
        "evidence_exact_set_match": evidence["exact_set_match"]["value"],
        "evidence_references_valid": record.get("evidence_references_valid"),
        "invalid_evidence_reference_count": evidence["invalid_reference_count"],
        "unresolved_evidence_references": json.dumps(record["unresolved_evidence_references"], ensure_ascii=False, sort_keys=True),
        "call_latency_ms": record.get("call_latency_ms"),
        "gold_verdict": compliance.get("gold_verdict"),
        "predicted_verdict": compliance.get("predicted_verdict"),
        "compliance_outcome_status": compliance.get("outcome_status"),
        "confusion_category": compliance.get("confusion_category"),
        "binary_correct": compliance.get("binary_correct"),
        "breach_outcome": compliance.get("breach_outcome"),
        "expected_value": extraction.get("expected_value"),
        "value_type": extraction.get("value_type"),
        "gold_value_status": extraction.get("gold_value_status"),
        "evaluator_support": extraction.get("evaluator_support"),
        "extraction_evidence_status": extraction.get("evidence_status"),
        "extraction_evidence_coverage": extraction.get("evidence_coverage"),
    }


def write_run_artifacts(
    run_dir: Path,
    *,
    metadata: dict[str, Any],
    raw_outputs: dict[str, Any],
    predictions: list[dict[str, Any]],
    scores: list[dict[str, Any]],
    summary: dict[str, Any],
    errors: list[dict[str, Any]],
) -> None:
    """Create a new run directory. Existing runs are never overwritten."""
    run_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir()
    _write_json(run_dir / "metadata.json", metadata)
    for call_id in sorted(raw_outputs):
        _write_json(raw_dir / f"{call_id}.json", raw_outputs[call_id])
    _write_jsonl(run_dir / "predictions.jsonl", predictions)
    _write_jsonl(run_dir / "scores.jsonl", scores)
    _write_json(run_dir / "summary.json", summary)
    _write_jsonl(run_dir / "errors.jsonl", errors)

    rows = [_csv_row(record) for record in scores]
    with (run_dir / "results.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
