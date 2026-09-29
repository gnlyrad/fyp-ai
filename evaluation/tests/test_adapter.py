from fyp_evaluation_engine.adapter import normalize_call_output


CHECKLIST = [
    {"id": "C01", "type": "COMPLIANCE", "requirement_type": "OBLIGATION", "question": "Required."},
    {"id": "K01", "type": "CLIENT_KYC", "field": "identifier", "question": "Identifier?"},
]


def test_explicit_null_verdict_failure_preserves_stage_diagnostics():
    diagnostics = [{"stage": "evidence_audit", "attempts": 3, "status": "prediction_failure"}]
    findings = [{"checklist_id": item["id"], "verdict": None,
                 "execution_status": "prediction_failure", "failure_category": "analysis_unavailable",
                 "evidence_status": "analysis_unavailable", "failure_stage": "evidence_audit",
                 "failure_error_type": "ClientError", "failure_error_code": "ThrottlingException",
                 "analysis_diagnostics": diagnostics, "spans": []} for item in CHECKLIST]
    predictions, errors = normalize_call_output("CALL_001", CHECKLIST, {1}, _output(findings), 1.0)
    assert all(p["execution_status"] == "prediction_failure" for p in predictions)
    assert all(p["failure_category"] == "analysis_unavailable" for p in predictions)
    assert all(p["native_verdict"] is None and p["predicted_verdict"] is None for p in predictions)
    assert predictions[0]["graph_metadata"]["finding"]["analysis_diagnostics"] == diagnostics
    assert errors[0]["details"]["failure_stage"] == "evidence_audit"
    assert errors[0]["details"]["failure_error_code"] == "ThrottlingException"


def test_applicability_service_failure_is_not_an_abstention():
    findings = [{"checklist_id": item["id"], "verdict": None,
                 "execution_status": "prediction_failure", "evidence_status": "applicability_unavailable",
                 "spans": []} for item in CHECKLIST]
    predictions, _ = normalize_call_output("CALL_001", CHECKLIST, {1}, _output(findings), 1.0)
    assert all(p["failure_category"] == "applicability_unavailable" for p in predictions)


def _output(findings, chunks=None):
    return {
        "chunks": chunks or [],
        "report": {"findings": findings, "summary": {}, "warnings": [], "consistency": {}},
    }


def test_maps_final_spans_via_source_segment_id_and_ignores_candidate_chunks():
    finding = {
        "checklist_id": "C01",
        "verdict": "Fail",
        "evidence_status": "direct_evidence",
        "reviewed_chunk_ids": ["chunk-00", "chunk-01"],
        "retrieved_evidence": [{"chunk_id": "chunk-00"}],
        "spans": [{"chunk_id": "chunk-01", "text": "decisive evidence"}],
    }
    missing = {"checklist_id": "K01", "verdict": "Pass", "evidence_status": "absence_after_full_review", "spans": []}
    chunks = [
        {"chunk_id": "chunk-00", "source_segment_id": 90},
        {"chunk_id": "chunk-01", "source_segment_id": 42},
    ]
    predictions, errors = normalize_call_output("CALL_001", CHECKLIST, {42, 90}, _output([finding, missing], chunks), 12.0)
    assert not errors
    assert predictions[0]["predicted_verdict"] == "FAIL"
    assert predictions[0]["predicted_evidence_turn_ids"] == [42]
    assert predictions[0]["evidence_references_valid"] is True
    assert predictions[1]["predicted_evidence_turn_ids"] == []
    assert predictions[1]["evidence_availability"] == "available"


def test_invalid_evidence_reference_is_preserved_as_diagnostic():
    findings = [
        {"checklist_id": "C01", "verdict": "Pass", "evidence_status": "direct_evidence",
         "spans": [{"chunk_id": "chunk-99", "text": "bad"}]},
        {"checklist_id": "K01", "verdict": "Pass", "evidence_status": "direct_evidence", "spans": []},
    ]
    predictions, errors = normalize_call_output("CALL_001", CHECKLIST, {1}, _output(findings), 1.0)
    assert predictions[0]["evidence_references_valid"] is False
    assert predictions[0]["unresolved_evidence_references"][0]["chunk_id"] == "chunk-99"
    assert any(error["category"] == "invalid_evidence_reference" for error in errors)


def test_does_not_parse_turn_id_from_chunk_name_when_source_mapping_is_missing():
    findings = [
        {"checklist_id": "C01", "verdict": "Pass", "evidence_status": "direct_evidence",
         "spans": [{"chunk_id": "chunk-04", "text": "quote"}]},
        {"checklist_id": "K01", "verdict": "Pass", "evidence_status": "direct_evidence", "spans": []},
    ]
    predictions, _ = normalize_call_output(
        "CALL_001", CHECKLIST, {4}, _output(findings, [{"chunk_id": "chunk-04"}]), 1.0
    )
    assert predictions[0]["predicted_evidence_turn_ids"] == []
    assert "no source_segment_id" in predictions[0]["unresolved_evidence_references"][0]["reason"]


def test_graph_failure_and_missing_prediction_are_not_abstentions():
    predictions, errors = normalize_call_output(
        "CALL_001", CHECKLIST, {1}, None, 3.0, "RuntimeError: unavailable"
    )
    assert all(prediction["execution_status"] == "prediction_failure" for prediction in predictions)
    assert all(prediction["failure_category"] == "missing_prediction" for prediction in predictions)
    assert all(prediction["predicted_verdict"] is None for prediction in predictions)
    assert [error["category"] for error in errors].count("graph_exception") == 1
    assert [error["category"] for error in errors].count("missing_prediction") == 2
    assert {error["evaluation_family"] for error in errors if error["question_id"]} == {"compliance", "extraction"}


def test_caught_analysis_failure_is_operational_failure_and_keeps_native_verdict():
    findings = [
        {"checklist_id": "C01", "verdict": "Unclear", "evidence_status": "analysis_unavailable", "spans": [],
         "reason": "Analysis or evidence verification unavailable: ValidationError."},
        {"checklist_id": "K01", "verdict": "Unclear", "evidence_status": "analysis_unavailable", "spans": []},
    ]
    predictions, errors = normalize_call_output("CALL_001", CHECKLIST, set(), _output(findings), 1.0)
    for prediction in predictions:
        assert prediction["native_verdict"] == "Unclear"
        assert prediction["predicted_verdict"] is None
        assert prediction["execution_status"] == "prediction_failure"
        assert prediction["failure_category"] == "analysis_unavailable"
    failures = [error for error in errors if error["category"] == "prediction_failure"]
    assert {(error["question_id"], error["evaluation_family"]) for error in failures} == {
        ("C01", "compliance"), ("K01", "extraction")}
    assert all(error["details"]["native_verdict"] == "Unclear" for error in failures)
    assert all(error["details"]["failure_category"] == "analysis_unavailable" for error in failures)
    assert "ValidationError" in failures[0]["details"]["graph_reason"]


def test_predictions_carry_family_and_only_compliance_gets_a_normalized_verdict():
    findings = [
        {"checklist_id": "C01", "verdict": "Unclear", "evidence_status": "absence_after_full_review", "spans": []},
        {"checklist_id": "K01", "verdict": "Unclear", "evidence_status": "absence_after_full_review", "spans": []},
    ]
    predictions, errors = normalize_call_output("CALL_001", CHECKLIST, set(), _output(findings), 1.0)
    compliance, extraction = predictions
    assert (compliance["evaluation_family"], compliance["predicted_verdict"]) == ("compliance", "UNCLEAR")
    assert (extraction["evaluation_family"], extraction["predicted_verdict"]) == ("extraction", None)
    assert extraction["native_verdict"] == "Unclear" and extraction["execution_status"] == "success"
    assert extraction["question_type"] == "CLIENT_KYC"
    assert not errors


def test_unrecognised_native_verdict_is_a_prediction_failure():
    findings = [
        {"checklist_id": "C01", "verdict": "Maybe", "evidence_status": "direct_evidence", "spans": []},
        {"checklist_id": "K01", "verdict": "Pass", "evidence_status": "direct_evidence", "spans": []},
    ]
    predictions, errors = normalize_call_output("CALL_001", CHECKLIST, set(), _output(findings), 1.0)
    assert predictions[0]["failure_category"] == "malformed_verdict"
    assert any(error["category"] == "malformed_verdict" for error in errors)
