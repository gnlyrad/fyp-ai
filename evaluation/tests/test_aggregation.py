import pytest
from helpers import score_compliance, score_extraction

from fyp_evaluation_engine.scoring import aggregate_scores


def _failed_compliance(gold, **kwargs):
    return score_compliance(
        gold, None, status="prediction_failure", native="Unclear",
        availability="unavailable", failure_category="analysis_unavailable", **kwargs
    )


def _records():
    return [
        # gold FAIL breakdown: detected, false clearance, abstained, failed, unexpected N/A
        score_compliance("FAIL", "FAIL", call_id="CALL_1", question_id="C01"),
        score_compliance("FAIL", "PASS", call_id="CALL_1", question_id="C02"),
        score_compliance("FAIL", "UNCLEAR", call_id="CALL_2", question_id="C01"),
        _failed_compliance("FAIL", call_id="CALL_2", question_id="C02"),
        score_compliance("FAIL", "NOT_APPLICABLE", call_id="CALL_3", question_id="C01"),
        # gold PASS: TN, FP, abstention, failure
        score_compliance("PASS", "PASS", call_id="CALL_3", question_id="C02"),
        score_compliance("PASS", "FAIL", call_id="CALL_4", question_id="C01"),
        score_compliance("PASS", "UNCLEAR", call_id="CALL_4", question_id="C02"),
        _failed_compliance("PASS", call_id="CALL_5", question_id="C01"),
        # extraction: FOUND, NOT_FOUND (successful native Unclear), failure on PRESENT,
        # NOT_STATED + failure, NOT_STATED success, UNSUPPORTED
        score_extraction("SGD", call_id="CALL_1", question_id="K01", evidence=[{"text": "SGD"}]),
        score_extraction("SGD", call_id="CALL_2", question_id="K01", native="Unclear", evidence=[{"text": "USD"}]),
        score_extraction("SGD", call_id="CALL_3", question_id="K01", native="Unclear", status="prediction_failure",
                         availability="unavailable"),
        score_extraction(None, call_id="CALL_4", question_id="K01", native="Unclear", status="prediction_failure",
                         availability="unavailable"),
        score_extraction(None, call_id="CALL_5", question_id="K01", evidence=[{"text": "whatever"}]),
        score_extraction(7, "integer", call_id="CALL_5", question_id="K02", evidence=[{"text": "7"}]),
    ]


@pytest.fixture(scope="module")
def summary():
    return aggregate_scores(_records())


def test_counts_and_native_unclear_reconcile(summary):
    counts = summary["overall"]["counts"]
    assert (counts["total_questions"], counts["compliance_questions"], counts["extraction_questions"]) == (15, 9, 6)
    # native "Unclear" = compliance abstentions + failure fallbacks + extraction outputs/failures
    reconciliation = counts["native_unclear_reconciliation"]
    assert reconciliation == {
        "total_native_unclear": 7,
        "compliance_abstentions": 2,
        "compliance_prediction_failures": 2,
        "extraction_successful_outputs": 1,
        "extraction_prediction_failures": 2,
    }
    assert reconciliation["total_native_unclear"] == sum(
        value for key, value in reconciliation.items() if key != "total_native_unclear"
    )
    assert counts["native_verdict"]["Unclear"] == reconciliation["total_native_unclear"]


def test_native_unclear_is_counted_for_compliance_from_the_raw_graph_value(summary):
    diagnostics = summary["overall"]["compliance"]["native_diagnostics"]
    assert diagnostics["native_unclear_outputs"]["numerator"] == 4
    assert diagnostics["native_unclear_successful"]["numerator"] == 2
    assert diagnostics["native_unclear_prediction_failures"]["numerator"] == 2
    assert diagnostics["native_verdict"]["Unclear"] == 4


def test_compliance_outcomes_are_exclusive_and_reconcile(summary):
    compliance = summary["overall"]["compliance"]
    population = compliance["outcome_population"]
    assert population == {
        "total": 9, "binary_decisions": 4, "abstentions": 2, "prediction_failures": 2,
        "unexpected_not_applicable": 1,
    }
    assert sum(value for key, value in population.items() if key != "total") == population["total"]
    assert compliance["decision_coverage"]["value"] == pytest.approx(4 / 9)
    assert compliance["abstention_rate"]["value"] == pytest.approx(2 / 9)
    assert compliance["prediction_failure_rate"]["value"] == pytest.approx(2 / 9)
    assert compliance["unexpected_not_applicable_rate"]["value"] == pytest.approx(1 / 9)


def test_binary_metrics_use_only_completed_decisions(summary):
    binary = summary["overall"]["compliance"]["binary"]
    assert binary["confusion_matrix"] == {"TP": 1, "TN": 1, "FP": 1, "FN": 1}
    assert binary["accuracy"] == {"value": 0.5, "percentage": 50.0, "numerator": 2, "denominator": 4}
    assert binary["breach_precision"]["value"] == 0.5
    assert binary["breach_recall"]["value"] == 0.5


def test_gold_breach_outcomes_partition_every_gold_fail(summary):
    breach = summary["overall"]["compliance"]["breach_outcomes"]
    assert breach["gold_fail_total"] == 5
    assert breach["counts"] == {
        "detected": 1, "false_clearance": 1, "abstained": 1, "prediction_failure": 1,
        "unexpected_not_applicable": 1,
    }
    assert sum(breach["counts"].values()) == breach["gold_fail_total"]
    assert breach["automatic_breach_detection_rate"] == {
        "value": 0.2, "percentage": 20.0, "numerator": 1, "denominator": 5}
    for name in ("false_clearance_rate", "abstention_rate", "prediction_failure_rate",
                 "unexpected_not_applicable_rate"):
        assert breach[name]["denominator"] == 5 and breach[name]["numerator"] == 1


def test_extraction_counts_keep_dataset_execution_and_evidence_dimensions_separate(summary):
    extraction = summary["overall"]["extraction"]
    assert extraction["counts"] == {
        "total": 6,
        "present_expected_values": 4,       # SGD x3 + integer 7
        "not_stated_expected_values": 2,
        "unsupported": 1,
        "coverage_population": 3,           # PRESENT and SUPPORTED
        "found": 1,
        "not_found": 1,
        "prediction_failures": 2,           # one PRESENT, one NOT_STATED
        "coverage_population_prediction_failures": 1,
    }
    counts = extraction["counts"]
    assert counts["found"] + counts["not_found"] + counts["coverage_population_prediction_failures"] == \
        counts["coverage_population"]
    assert counts["coverage_population"] + counts["unsupported"] == counts["present_expected_values"]


def test_extraction_coverage_excludes_not_stated_and_unsupported_but_not_failures(summary):
    extraction = summary["overall"]["extraction"]
    assert extraction["extraction_evidence_coverage"] == {
        "value": pytest.approx(1 / 3), "percentage": pytest.approx(100 / 3), "numerator": 1, "denominator": 3}
    assert extraction["prediction_failure_rate"]["value"] == pytest.approx(2 / 6)


def test_extraction_reports_no_compliance_style_metrics(summary):
    extraction = summary["overall"]["extraction"]
    assert not {"binary", "abstention_rate", "decision_coverage", "breach_outcomes"} & set(extraction)


def test_family_specific_evidence_aggregates_differ_from_the_blended_one(summary):
    overall = summary["overall"]
    assert overall["compliance"]["evidence"]["overall"]["case_count"] == 9
    assert overall["extraction"]["evidence"]["overall"]["case_count"] == 6
    assert overall["evidence_all_questions"]["overall"]["case_count"] == 15
    assert overall["compliance"]["evidence"]["status_counts"] == {"prediction_failure": 2, "scored": 7}
    empty = overall["compliance"]["evidence"]["empty_gold"]
    assert empty["case_count"] == 9 and overall["compliance"]["evidence"]["non_empty_gold"]["case_count"] == 0


def test_by_call_and_by_question_use_only_applicable_family_blocks(summary):
    assert set(summary["by_call_id"]) == {"CALL_1", "CALL_2", "CALL_3", "CALL_4", "CALL_5"}
    assert set(summary["by_question_id"]) == {"C01", "C02", "K01", "K02"}
    assert summary["by_question_id"]["C01"]["extraction"] is None
    assert summary["by_question_id"]["K01"]["compliance"] is None
    assert summary["by_question_id"]["K02"]["extraction"]["counts"]["unsupported"] == 1
    assert summary["by_call_id"]["CALL_5"]["extraction"]["counts"]["total"] == 2


def test_undefined_ratios_are_null_not_zero():
    only_pass = aggregate_scores([score_compliance("PASS", "PASS")])["overall"]["compliance"]
    assert only_pass["binary"]["breach_recall"]["value"] is None
    assert only_pass["breach_outcomes"]["automatic_breach_detection_rate"]["value"] is None
    assert only_pass["binary"]["breach_recall"]["denominator"] == 0
    no_extraction_population = aggregate_scores([score_extraction(None)])["overall"]["extraction"]
    assert no_extraction_population["extraction_evidence_coverage"]["value"] is None
