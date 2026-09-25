import csv
import json
from pathlib import Path

import pytest
from helpers import score_compliance, score_extraction

from fyp_evaluation_engine.persistence import CSV_COLUMNS, write_run_artifacts

COMPLIANCE_ONLY = {"gold_verdict", "predicted_verdict", "compliance_outcome_status", "confusion_category",
                   "binary_correct", "breach_outcome"}
EXTRACTION_ONLY = {"expected_value", "value_type", "gold_value_status", "evaluator_support",
                   "extraction_evidence_status", "extraction_evidence_coverage"}


def _scores():
    return [
        score_compliance("FAIL", "FAIL", question_id="C01", turns=[1], evidence=[{"text": "x"}]),
        score_extraction("SGD", question_id="K01", evidence=[{"text": "SGD"}]),
        score_extraction(None, question_id="K02", status="prediction_failure", native="Unclear"),
    ]


def _write(path: Path):
    write_run_artifacts(
        path, metadata={"b": 2, "a": 1}, raw_outputs={"CALL_001": {"z": 1, "a": [2]}},
        predictions=[{"z": 2, "a": 1}], scores=_scores(), summary={"overall": {}}, errors=[],
    )


def test_persistent_serialization_is_stable_and_complete(tmp_path):
    first, second = tmp_path / "run-1", tmp_path / "run-2"
    _write(first)
    _write(second)
    expected = {"metadata.json", "raw", "predictions.jsonl", "scores.jsonl",
                "summary.json", "results.csv", "errors.jsonl"}
    assert {path.name for path in first.iterdir()} == expected
    assert (first / "raw" / "CALL_001.json").read_bytes() == (second / "raw" / "CALL_001.json").read_bytes()
    assert (first / "predictions.jsonl").read_bytes() == (second / "predictions.jsonl").read_bytes()
    assert (first / "results.csv").read_bytes() == (second / "results.csv").read_bytes()
    with pytest.raises(FileExistsError):
        _write(first)


def test_single_results_csv_supports_both_families_with_blanks_not_zeros(tmp_path):
    _write(tmp_path / "run")
    with (tmp_path / "run" / "results.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert tuple(reader.fieldnames) == CSV_COLUMNS
    assert [row["evaluation_family"] for row in rows] == ["compliance", "extraction", "extraction"]
    compliance, found, failed = rows
    assert all(compliance[name] == "" for name in EXTRACTION_ONLY)
    assert all(found[name] == "" for name in COMPLIANCE_ONLY)
    assert compliance["compliance_outcome_status"] == "binary_decision"
    assert compliance["breach_outcome"] == "detected"
    assert (found["gold_value_status"], found["extraction_evidence_status"], found["extraction_evidence_coverage"]) == (
        "PRESENT", "FOUND", "1")
    # a NOT_STATED prediction failure has no coverage value and no fake zero
    assert (failed["gold_value_status"], failed["execution_status"], failed["native_verdict"]) == (
        "NOT_STATED", "prediction_failure", "Unclear")
    assert failed["extraction_evidence_status"] == "" and failed["extraction_evidence_coverage"] == ""
    assert failed["evaluator_support"] == ""


def test_score_records_carry_what_a_read_only_ui_needs(tmp_path):
    _write(tmp_path / "run")
    records = [json.loads(line) for line in (tmp_path / "run" / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    shared = {"call_id", "question_id", "question_type", "evaluation_family", "execution_status",
              "failure_category", "native_verdict", "gold_evidence_turn_ids", "predicted_evidence_turn_ids",
              "evidence_references_valid", "evidence"}
    for record in records:
        assert shared <= set(record)
    compliance, extraction, _ = records
    assert extraction["compliance"] is None and compliance["extraction"] is None
    assert {"gold_verdict", "predicted_verdict", "outcome_status", "confusion_category", "binary_correct",
            "breach_outcome"} == set(compliance["compliance"])
    assert {"expected_value", "value_type", "gold_value_status", "evaluator_support", "evidence_status",
            "evidence_coverage"} == set(extraction["extraction"])
