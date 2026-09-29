"""Synthetic prediction/gold builders shared by the deterministic tests."""
from fyp_evaluation_engine.scoring import score_prediction

NATIVE = {"PASS": "Pass", "FAIL": "Fail", "UNCLEAR": "Unclear", "NOT_APPLICABLE": "N/A"}


def compliance_prediction(
    verdict="PASS", *, call_id="CALL_001", question_id="C01", status="success",
    native="auto", turns=(), evidence=(), availability="available", invalid=(),
    failure_category=None,
):
    """A normalized compliance prediction. ``verdict`` is the normalized verdict when successful."""
    if native == "auto":
        native = NATIVE.get(verdict)
    return {
        "call_id": call_id, "question_id": question_id, "question_type": "COMPLIANCE",
        "evaluation_family": "compliance", "requirement_type": "OBLIGATION", "field": None,
        "predicted_verdict": verdict if status == "success" else None, "native_verdict": native,
        "execution_status": status, "failure_category": failure_category,
        "predicted_evidence_turn_ids": list(turns), "selected_evidence": list(evidence),
        "unresolved_evidence_references": [],
        "evidence_reference_keys": [*(f"turn:{turn}" for turn in turns), *invalid],
        "evidence_references_valid": not invalid, "evidence_availability": availability,
        "graph_metadata": {}, "call_latency_ms": 1.0,
    }


def compliance_gold(verdict="PASS", turns=()):
    return {"expected": {"kind": "compliance", "expected_verdict": verdict}, "gold_evidence_turns": list(turns)}


def extraction_prediction(
    *, call_id="CALL_001", question_id="K01", question_type="CLIENT_KYC", status="success",
    native="Pass", turns=(), evidence=(), availability="available", invalid=(), failure_category=None,
):
    return {
        "call_id": call_id, "question_id": question_id, "question_type": question_type,
        "evaluation_family": "extraction", "requirement_type": None, "field": "field",
        "predicted_verdict": None, "native_verdict": native,
        "execution_status": status, "failure_category": failure_category,
        "predicted_evidence_turn_ids": list(turns), "selected_evidence": list(evidence),
        "unresolved_evidence_references": [],
        "evidence_reference_keys": [*(f"turn:{turn}" for turn in turns), *invalid],
        "evidence_references_valid": not invalid, "evidence_availability": availability,
        "graph_metadata": {}, "call_latency_ms": 1.0,
    }


def extraction_gold(value, value_type="text", turns=()):
    return {
        "expected": {"kind": "extraction", "expected_value": value, "value_type": value_type},
        "gold_evidence_turns": list(turns),
    }


def score_compliance(gold, verdict, **kwargs):
    return score_prediction(compliance_prediction(verdict, **kwargs), compliance_gold(gold))


def score_extraction(value, value_type="text", *, gold_turns=(), **kwargs):
    return score_prediction(extraction_prediction(**kwargs), extraction_gold(value, value_type, gold_turns))
