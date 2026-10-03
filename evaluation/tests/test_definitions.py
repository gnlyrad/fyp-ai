import json
from pathlib import Path

import pytest
from helpers import score_compliance, score_extraction

from fyp_evaluation_engine import METRIC_DEFINITIONS_VERSION
from fyp_evaluation_engine.scoring import BREACH_OUTCOMES, COMPLIANCE_OUTCOMES, aggregate_scores

DEFINITIONS = json.loads((Path(__file__).resolve().parents[1] / "metric_definitions.json").read_text(encoding="utf-8"))
REQUIRED = {"id", "label", "role", "applies_to", "population", "summary_path", "formula", "interpretation"}


def _resolve(node, path):
    for part in path.split("."):
        assert isinstance(node, dict) and part in node, f"missing {part!r} in {path}"
        node = node[part]
    return node


@pytest.fixture(scope="module")
def summary():
    return aggregate_scores([
        score_compliance("FAIL", "FAIL", question_id="C01"),
        score_extraction("SGD", question_id="K01", evidence=[{"text": "SGD"}]),
    ])


def test_definitions_version_matches_the_scorer():
    assert DEFINITIONS["schema_version"] == METRIC_DEFINITIONS_VERSION


def test_every_metric_is_fully_described_and_unique():
    ids = [metric["id"] for metric in DEFINITIONS["metrics"]]
    assert len(ids) == len(set(ids))
    for metric in DEFINITIONS["metrics"]:
        assert REQUIRED <= set(metric), metric["id"]
        assert metric["applies_to"] in {"compliance", "extraction", "both"}
        assert metric["role"] in {"headline", "secondary", "count", "diagnostic"}


def test_compliance_evidence_metrics_are_secondary_diagnostics():
    metrics = {metric["id"]: metric for metric in DEFINITIONS["metrics"]}
    for metric_id in (
        "compliance.evidence_precision",
        "compliance.evidence_recall",
        "compliance.evidence_exact_set_match",
    ):
        assert metrics[metric_id]["role"] == "secondary"


def test_every_defined_summary_path_exists_in_the_persisted_summary(summary):
    for metric in DEFINITIONS["metrics"]:
        paths = [metric["summary_path"], *metric.get("breakdowns", {}).values()]
        for path in paths:
            for node in (summary["overall"], summary["by_call_id"]["CALL_001"]):
                assert _resolve(node, path) is not None, metric["id"]


def test_every_persisted_headline_metric_has_a_definition(summary):
    defined = {metric["summary_path"] for metric in DEFINITIONS["metrics"]}
    headline_paths = [
        "compliance.decision_coverage", "compliance.abstention_rate", "compliance.prediction_failure_rate",
        "compliance.binary.accuracy", "compliance.binary.breach_precision", "compliance.binary.breach_recall",
        "compliance.binary.confusion_matrix",
        "compliance.breach_outcomes.automatic_breach_detection_rate",
        "compliance.breach_outcomes.false_clearance_rate", "compliance.breach_outcomes.abstention_rate",
        "compliance.breach_outcomes.prediction_failure_rate",
        "compliance.evidence.overall.precision", "compliance.evidence.overall.recall",
        "compliance.evidence.overall.exact_set_match",
        "extraction.extraction_evidence_coverage", "extraction.prediction_failure_rate",
        "extraction.evidence.overall.precision", "extraction.evidence.overall.recall",
        "extraction.evidence.overall.exact_set_match",
    ]
    assert set(headline_paths) <= defined
    for path in headline_paths:
        _resolve(summary["overall"], path)


def test_status_definitions_cover_every_implemented_state():
    groups = {group["id"]: {value["id"] for value in group["values"]} for group in DEFINITIONS["status_definitions"]}
    assert groups["execution_status"] == {"success", "prediction_failure"}
    assert groups["compliance_outcome_status"] == set(COMPLIANCE_OUTCOMES)
    assert groups["breach_outcome"] == set(BREACH_OUTCOMES)
    assert groups["extraction_gold_value_status"] == {"PRESENT", "NOT_STATED"}
    assert groups["extraction_evidence_status"] == {"FOUND", "NOT_FOUND"}
    assert groups["evaluator_support"] == {"SUPPORTED", "UNSUPPORTED"}
    assert groups["native_verdict"] == {"Pass", "Fail", "Unclear", "N/A"}
    assert {"scored", "invalid_references", "prediction_failure", "evidence_failure"} == groups["evidence_status"]


def test_scored_records_only_use_defined_statuses():
    groups = {group["id"]: {value["id"] for value in group["values"]} for group in DEFINITIONS["status_definitions"]}
    records = [
        score_compliance("FAIL", "FAIL"), score_compliance("FAIL", "UNCLEAR"),
        score_compliance("FAIL", None, status="prediction_failure", failure_category="analysis_unavailable"),
        score_extraction("A", evidence=[{"text": "A"}]), score_extraction("A", evidence=[{"text": "B"}]),
        score_extraction(None), score_extraction(1, "integer"),
    ]
    for record in records:
        assert record["execution_status"] in groups["execution_status"]
        assert record["evidence"]["status"] in groups["evidence_status"]
        if record["failure_category"]:
            assert record["failure_category"] in groups["failure_category"]
        if record["compliance"]:
            assert record["compliance"]["outcome_status"] in groups["compliance_outcome_status"]
        if record["extraction"]:
            extraction = record["extraction"]
            assert extraction["gold_value_status"] in groups["extraction_gold_value_status"]
            assert extraction["evidence_status"] in groups["extraction_evidence_status"] | {None}
            assert extraction["evaluator_support"] in groups["evaluator_support"] | {None}
