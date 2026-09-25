import pytest
from helpers import (
    compliance_gold,
    compliance_prediction,
    extraction_gold,
    extraction_prediction,
    score_compliance,
    score_extraction,
)

from fyp_evaluation_engine.scoring import score_evidence_sets, score_prediction

# ---- family classification -------------------------------------------------


@pytest.mark.parametrize(
    ("question_type", "family"),
    [("COMPLIANCE", "compliance"), ("CLIENT_KYC", "extraction"), ("TRANSACTION", "extraction")],
)
def test_family_is_derived_from_question_type_metadata_not_question_id(question_type, family):
    prediction = compliance_prediction("PASS", question_id="Z99")
    prediction.pop("evaluation_family")
    prediction["question_type"] = question_type
    gold = compliance_gold() if family == "compliance" else extraction_gold("x")
    record = score_prediction(prediction, gold)
    assert record["evaluation_family"] == family
    assert (record["compliance"] is None) == (family == "extraction")
    assert (record["extraction"] is None) == (family == "compliance")


def test_family_must_agree_with_ground_truth_kind():
    with pytest.raises(ValueError, match="does not match evaluation family"):
        score_prediction(compliance_prediction("PASS"), extraction_gold("x"))


# ---- compliance outcomes ---------------------------------------------------


@pytest.mark.parametrize(
    ("gold", "predicted", "category", "correct"),
    [("FAIL", "FAIL", "TP", 1), ("PASS", "PASS", "TN", 1),
     ("PASS", "FAIL", "FP", 0), ("FAIL", "PASS", "FN", 0)],
)
def test_successful_pass_or_fail_is_a_binary_decision(gold, predicted, category, correct):
    compliance = score_compliance(gold, predicted)["compliance"]
    assert compliance["outcome_status"] == "binary_decision"
    assert compliance["confusion_category"] == category
    assert compliance["binary_correct"] == correct


def test_successful_unclear_is_an_abstention():
    compliance = score_compliance("FAIL", "UNCLEAR")["compliance"]
    assert compliance["outcome_status"] == "abstention"
    assert compliance["confusion_category"] is None and compliance["binary_correct"] is None
    assert compliance["breach_outcome"] == "abstained"


def test_unclear_after_analysis_failure_is_a_prediction_failure_not_an_abstention():
    record = score_compliance(
        "FAIL", "UNCLEAR", status="prediction_failure", native="Unclear", failure_category="analysis_unavailable"
    )
    assert record["execution_status"] == "prediction_failure"
    assert record["failure_category"] == "analysis_unavailable"
    assert record["native_verdict"] == "Unclear"  # preserved for diagnostics
    assert record["compliance"]["outcome_status"] == "prediction_failure"
    assert record["compliance"]["predicted_verdict"] is None
    assert record["compliance"]["breach_outcome"] == "prediction_failure"


def test_not_applicable_is_unexpected_because_applicability_is_bypassed():
    record = score_compliance("FAIL", "NOT_APPLICABLE")
    assert record["compliance"]["outcome_status"] == "unexpected_not_applicable"
    assert record["compliance"]["breach_outcome"] == "unexpected_not_applicable"


def test_breach_outcome_only_exists_for_gold_fail():
    assert score_compliance("PASS", "FAIL")["compliance"]["breach_outcome"] is None
    assert score_compliance("FAIL", "FAIL")["compliance"]["breach_outcome"] == "detected"
    assert score_compliance("FAIL", "PASS")["compliance"]["breach_outcome"] == "false_clearance"


def test_legacy_granular_execution_statuses_collapse_to_prediction_failure():
    prediction = compliance_prediction(None, status="missing_prediction", availability="unavailable")
    prediction.pop("failure_category")
    record = score_prediction(prediction, compliance_gold("FAIL"))
    assert record["execution_status"] == "prediction_failure"
    assert record["failure_category"] == "missing_prediction"


# ---- native UNCLEAR semantics ---------------------------------------------


def test_native_unclear_alone_never_decides_the_evaluation_status():
    successful_compliance = score_compliance("PASS", "UNCLEAR")
    failed_compliance = score_compliance("PASS", None, status="prediction_failure", native="Unclear")
    successful_extraction = score_extraction("ABC", native="Unclear", evidence=[{"text": "no value"}])
    failed_extraction = score_extraction("ABC", native="Unclear", status="prediction_failure")
    assert successful_compliance["compliance"]["outcome_status"] == "abstention"
    assert failed_compliance["compliance"]["outcome_status"] == "prediction_failure"
    assert successful_extraction["compliance"] is None  # never a compliance abstention
    assert successful_extraction["extraction"]["evidence_status"] == "NOT_FOUND"
    assert failed_extraction["extraction"]["evidence_status"] is None


# ---- evidence sets ---------------------------------------------------------


@pytest.mark.parametrize(
    ("gold", "predicted", "expected"),
    [([1], [1], (1, 1, 1)), ([1], [], (0, 0, 0)),
     ([], [1], (0, 0, 0)), ([], [], (1, 1, 1))],
)
def test_custom_empty_set_convention(gold, predicted, expected):
    score = score_evidence_sets(gold, predicted)
    actual = tuple(score[name]["value"] for name in ("precision", "recall", "exact_set_match"))
    assert actual == expected


def test_multiple_gold_and_predicted_evidence_turns():
    score = score_evidence_sets([1, 2], [1, 2, 3])
    assert score["precision"]["value"] == pytest.approx(2 / 3)
    assert score["recall"]["value"] == 1
    assert score["exact_set_match"]["value"] == 0


def test_invalid_reference_is_in_precision_denominator_and_blocks_exact_match():
    score = score_evidence_sets([1], [1], invalid_reference_keys=["invalid:chunk-99:0"])
    assert score["status"] == "invalid_references"
    assert score["precision"]["value"] == 0.5
    assert score["recall"]["value"] == 1
    assert score["exact_set_match"]["value"] == 0


def test_failed_evidence_gets_zero_not_empty_empty_credit():
    score = score_evidence_sets([], [], available=False)
    assert score["status"] == "evidence_failure"
    assert [score[name]["value"] for name in ("precision", "recall", "exact_set_match")] == [0, 0, 0]


def test_prediction_failure_evidence_is_not_rewarded_as_legitimate_empty_evidence():
    record = score_compliance("PASS", None, status="prediction_failure", native="Unclear", availability="unavailable")
    assert record["evidence"]["status"] == "prediction_failure"
    assert record["evidence"]["exact_set_match"]["value"] == 0


def test_extraction_evidence_diagnostics_are_independent_of_found_status():
    record = score_extraction(
        "SGD", gold_turns=[17], turns=[16, 17, 18], native="Unclear",
        evidence=[{"text": "The currency is SGD."}],
    )
    assert record["extraction"]["evidence_status"] == "FOUND"
    assert record["evidence"]["recall"]["value"] == 1
    assert record["evidence"]["precision"]["value"] == pytest.approx(1 / 3)
    assert record["evidence"]["exact_set_match"]["value"] == 0


# ---- extraction ------------------------------------------------------------


def test_present_value_in_final_evidence_is_found():
    extraction = score_extraction("SGD", evidence=[{"text": "in SGD"}])["extraction"]
    assert (extraction["gold_value_status"], extraction["evidence_status"], extraction["evidence_coverage"]) == (
        "PRESENT", "FOUND", 1)
    assert extraction["evaluator_support"] == "SUPPORTED"


def test_present_value_missing_from_final_evidence_is_not_found():
    extraction = score_extraction("SGD", evidence=[{"text": "in USD"}])["extraction"]
    assert (extraction["evidence_status"], extraction["evidence_coverage"]) == ("NOT_FOUND", 0)


def test_prediction_failure_has_no_found_status_but_stays_in_the_coverage_population():
    record = score_extraction("SGD", status="prediction_failure", native="Unclear", availability="unavailable")
    assert record["extraction"]["evidence_status"] is None
    assert record["extraction"]["evidence_coverage"] == 0
    assert record["extraction"]["gold_value_status"] == "PRESENT"


def test_not_stated_is_a_gold_property_outside_the_coverage_population():
    for kwargs in ({"evidence": [{"text": "null NOT_STATED"}]}, {"status": "prediction_failure"}):
        extraction = score_extraction(None, **kwargs)["extraction"]
        assert extraction["gold_value_status"] == "NOT_STATED"
        assert extraction["evidence_status"] is None
        assert extraction["evidence_coverage"] is None
        assert extraction["evaluator_support"] is None


def test_text_uses_token_boundaries_not_naive_substrings():
    assert score_extraction("100", evidence=[{"text": "The amount is 1000."}])["extraction"]["evidence_status"] == "NOT_FOUND"
    assert score_extraction("100", evidence=[{"text": "The amount is 100."}])["extraction"]["evidence_status"] == "FOUND"


def test_date_normalizes_leading_zero():
    extraction = score_extraction(
        "04 August 1991", "dob_dd_month_yyyy", evidence=[{"text": "Born 4 August 1991."}]
    )["extraction"]
    assert extraction["evidence_status"] == "FOUND"


def test_unsupported_only_for_an_unsupported_value_type_or_contract():
    integer = score_extraction(123, "integer", evidence=[{"text": "123"}])["extraction"]
    bad_date = score_extraction("not a date", "dob_dd_month_yyyy", evidence=[{"text": "x"}])["extraction"]
    for extraction in (integer, bad_date):
        assert extraction["evaluator_support"] == "UNSUPPORTED"
        assert extraction["evidence_status"] is None
        assert extraction["evidence_coverage"] is None


def test_invalid_evidence_reference_is_not_unsupported():
    record = score_extraction(
        "SGD", turns=[3], invalid=["invalid:chunk-99:1"], availability="invalid_references",
        evidence=[{"text": "SGD"}, {"text": "bogus"}],
    )
    assert record["extraction"]["evaluator_support"] == "SUPPORTED"
    assert record["evidence"]["status"] == "invalid_references"
    assert record["evidence"]["invalid_reference_count"] == 1


def test_successful_native_unclear_extraction_can_be_found_or_not_found():
    found = score_extraction("SGD", native="Unclear", evidence=[{"text": "SGD"}])
    not_found = score_extraction("SGD", native="Unclear", evidence=[{"text": "USD"}])
    assert found["extraction"]["evidence_status"] == "FOUND"
    assert not_found["extraction"]["evidence_status"] == "NOT_FOUND"
    assert found["native_verdict"] == not_found["native_verdict"] == "Unclear"


def test_extraction_record_has_no_compliance_fields():
    record = score_extraction("SGD", evidence=[{"text": "SGD"}])
    assert record["compliance"] is None
    assert "answer" not in record
