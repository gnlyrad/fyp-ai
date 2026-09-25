"""Deterministic compliance, extraction, evidence, and aggregation metrics.

Every call-question belongs to exactly one evaluation family (``compliance`` or
``extraction``) and is scored with that family's contract only.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Any, Iterable

from fyp_evaluation_loader import evaluation_family_for_question_type

COMPLIANCE_OUTCOMES = ("binary_decision", "abstention", "prediction_failure", "unexpected_not_applicable")
BREACH_OUTCOMES = ("detected", "false_clearance", "abstained", "prediction_failure", "unexpected_not_applicable")
TEXT_VALUE_TYPE = "text"
DATE_VALUE_TYPE = "dob_dd_month_yyyy"


def _ratio(numerator: int | float, denominator: int | float) -> dict[str, Any]:
    value = numerator / denominator if denominator else None
    return {
        "value": value,
        "percentage": value * 100 if value is not None else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def score_evidence_sets(
    gold_turn_ids: Iterable[Any],
    predicted_turn_ids: Iterable[Any],
    *,
    available: bool = True,
    invalid_reference_keys: Iterable[str] = (),
    failure_status: str = "evidence_failure",
) -> dict[str, Any]:
    """Score evidence sets with the project's explicit empty-set convention.

    Unavailable evidence receives a zero failure penalty; it is not interpreted
    as a legitimate empty prediction. Invalid references remain in the predicted
    precision denominator and make exact match impossible.
    """
    gold = {f"turn:{value}" for value in gold_turn_ids}
    predicted = {f"turn:{value}" for value in predicted_turn_ids}
    invalid = set(invalid_reference_keys)
    if not available:
        return {
            "status": failure_status,
            "gold_set_size": len(gold),
            "predicted_set_size": None,
            "invalid_reference_count": len(invalid),
            "intersection_size": None,
            "precision": _ratio(0, 1),
            "recall": _ratio(0, 1),
            "exact_set_match": _ratio(0, 1),
        }

    predicted_all = predicted | invalid
    intersection = gold & predicted
    if not gold and not predicted_all:
        precision = recall = exact = _ratio(1, 1)
    elif not predicted_all:
        precision, recall, exact = _ratio(0, 1), _ratio(0, len(gold)), _ratio(0, 1)
    elif not gold:
        precision, recall, exact = _ratio(0, len(predicted_all)), _ratio(0, 1), _ratio(0, 1)
    else:
        precision = _ratio(len(intersection), len(predicted_all))
        recall = _ratio(len(intersection), len(gold))
        exact = _ratio(int(gold == predicted_all), 1)
    return {
        "status": "invalid_references" if invalid else "scored",
        "gold_set_size": len(gold),
        "predicted_set_size": len(predicted_all),
        "invalid_reference_count": len(invalid),
        "intersection_size": len(intersection),
        "precision": precision,
        "recall": recall,
        "exact_set_match": exact,
    }


def _execution(prediction: dict[str, Any]) -> tuple[str, str | None]:
    """Collapse to success / prediction_failure; also accepts pre-v2 granular statuses."""
    status = prediction.get("execution_status")
    if status == "success":
        return "success", None
    category = prediction.get("failure_category")
    if category is None and status not in (None, "prediction_failure"):
        category = status  # legacy: missing_prediction / malformed_prediction
    return "prediction_failure", category


def _family(prediction: dict[str, Any], gold: dict[str, Any]) -> str:
    family = prediction.get("evaluation_family") or evaluation_family_for_question_type(prediction["question_type"])
    if gold["expected"].get("kind") != family:
        raise ValueError(
            f"{prediction['call_id']}/{prediction['question_id']}: ground-truth kind "
            f"{gold['expected'].get('kind')!r} does not match evaluation family {family!r}"
        )
    return family


def _compliance_score(prediction: dict[str, Any], gold: dict[str, Any], execution: str) -> dict[str, Any]:
    expected = gold["expected"]["expected_verdict"]
    predicted = prediction.get("predicted_verdict") if execution == "success" else None
    category = binary_correct = None
    if execution != "success":
        outcome = "prediction_failure"
    elif predicted == "UNCLEAR":
        outcome = "abstention"
    elif predicted == "NOT_APPLICABLE":
        outcome = "unexpected_not_applicable"
    elif predicted in {"PASS", "FAIL"}:
        outcome = "binary_decision"
        category = {
            ("FAIL", "FAIL"): "TP", ("PASS", "PASS"): "TN",
            ("PASS", "FAIL"): "FP", ("FAIL", "PASS"): "FN",
        }[(expected, predicted)]
        binary_correct = int(category in {"TP", "TN"})
    else:
        outcome = "prediction_failure"
    breach_outcome = None
    if expected == "FAIL":
        breach_outcome = {
            "binary_decision": "detected" if category == "TP" else "false_clearance",
            "abstention": "abstained",
            "prediction_failure": "prediction_failure",
            "unexpected_not_applicable": "unexpected_not_applicable",
        }[outcome]
    return {
        "gold_verdict": expected,
        "predicted_verdict": predicted,
        "outcome_status": outcome,
        "confusion_category": category,
        "binary_correct": binary_correct,
        "breach_outcome": breach_outcome,
    }


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.findall(r"[^\W_]+", normalized, flags=re.UNICODE)


def _contains_token_sequence(expected: str, evidence: str) -> bool:
    needle, haystack = _tokens(expected), _tokens(evidence)
    if not needle:
        return False
    width = len(needle)
    return any(haystack[index:index + width] == needle for index in range(len(haystack) - width + 1))


_MONTHS = "January February March April May June July August September October November December"
_DATE_PATTERN = re.compile(
    rf"(?<!\w)(\d{{1,2}})\s+({_MONTHS.replace(' ', '|')})\s+(\d{{4}})(?!\w)",
    flags=re.IGNORECASE,
)


def _parse_expected_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.strip(), "%d %B %Y").date().isoformat()
    except ValueError:
        return None


def _evidence_dates(value: str) -> set[str]:
    dates = set()
    for match in _DATE_PATTERN.finditer(unicodedata.normalize("NFKC", value)):
        try:
            dates.add(datetime.strptime(" ".join(match.groups()), "%d %B %Y").date().isoformat())
        except ValueError:
            continue
    return dates


def _evaluator_support(value: Any, value_type: Any) -> str:
    """SUPPORTED when a deterministic comparison exists for this expected value."""
    if value_type == TEXT_VALUE_TYPE and isinstance(value, str) and _tokens(value):
        return "SUPPORTED"
    if value_type == DATE_VALUE_TYPE and _parse_expected_date(value) is not None:
        return "SUPPORTED"
    return "UNSUPPORTED"


def _extraction_score(prediction: dict[str, Any], gold: dict[str, Any], execution: str) -> dict[str, Any]:
    """Score whether the final selected evidence contains the expected value.

    ``gold_value_status`` describes the dataset (PRESENT / NOT_STATED).
    ``evidence_status`` (FOUND / NOT_FOUND) exists only for a successful, PRESENT,
    SUPPORTED row. ``evidence_coverage`` is the per-row headline contribution: 1 if
    FOUND, 0 if the row is in the coverage population but not FOUND (including a
    prediction failure), null when the row is outside it (NOT_STATED or UNSUPPORTED).
    """
    expected = gold["expected"]
    value, value_type = expected.get("expected_value"), expected.get("value_type")
    support = _evaluator_support(value, value_type) if value is not None else None
    evidence_status = coverage = None
    if support == "SUPPORTED":
        coverage = 0
        if execution == "success":
            evidence = "\n".join(
                span["text"] for span in prediction.get("selected_evidence", [])
                if isinstance(span, dict) and isinstance(span.get("text"), str)
            )
            if value_type == TEXT_VALUE_TYPE:
                found = _contains_token_sequence(value, evidence)
            else:
                found = _parse_expected_date(value) in _evidence_dates(evidence)
            evidence_status = "FOUND" if found else "NOT_FOUND"
            coverage = int(found)
    return {
        "expected_value": value,
        "value_type": value_type,
        "gold_value_status": "NOT_STATED" if value is None else "PRESENT",
        "evaluator_support": support,
        "evidence_status": evidence_status,
        "evidence_coverage": coverage,
    }


def score_prediction(prediction: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """Create a self-contained, family-specific score record for later aggregation."""
    family = _family(prediction, gold)
    execution, failure_category = _execution(prediction)
    invalid_keys = [key for key in prediction.get("evidence_reference_keys", [])
                    if isinstance(key, str) and key.startswith("invalid:")]
    evidence_available = (
        execution == "success"
        and prediction.get("evidence_availability") in {"available", "invalid_references"}
    )
    evidence = score_evidence_sets(
        gold.get("gold_evidence_turns", []),
        prediction.get("predicted_evidence_turn_ids", []),
        available=evidence_available,
        invalid_reference_keys=invalid_keys,
        failure_status="prediction_failure" if execution != "success" else "evidence_failure",
    )
    return {
        "call_id": prediction["call_id"],
        "question_id": prediction["question_id"],
        "question_type": prediction["question_type"],
        "evaluation_family": family,
        "requirement_type": prediction.get("requirement_type"),
        "field": prediction.get("field"),
        "execution_status": execution,
        "failure_category": failure_category,
        "native_verdict": prediction.get("native_verdict"),
        "call_latency_ms": prediction.get("call_latency_ms"),
        "gold_evidence_turn_ids": list(gold.get("gold_evidence_turns", [])),
        "predicted_evidence_turn_ids": list(prediction.get("predicted_evidence_turn_ids", [])),
        "unresolved_evidence_references": list(prediction.get("unresolved_evidence_references", [])),
        "evidence_references_valid": prediction.get("evidence_references_valid"),
        "evidence_availability": prediction.get("evidence_availability"),
        "evidence": evidence,
        "compliance": _compliance_score(prediction, gold, execution) if family == "compliance" else None,
        "extraction": _extraction_score(prediction, gold, execution) if family == "extraction" else None,
    }


def _is_native_unclear(record: dict[str, Any]) -> bool:
    """The graph emits ``Unclear``; compare its raw value, never the normalized ``UNCLEAR``."""
    native = record.get("native_verdict")
    return isinstance(native, str) and native.strip().casefold() == "unclear"


def _mean_metric(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    values = [record["evidence"][name]["value"] for record in records]
    return _ratio(sum(values), len(values))


def _evidence_block(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(items),
        "precision": _mean_metric(items, "precision"),
        "recall": _mean_metric(items, "recall"),
        "exact_set_match": _mean_metric(items, "exact_set_match"),
    }


def _evidence_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "aggregation": "macro_mean_of_per_question_scores",
        "empty_set_convention": "project_custom",
        "overall": _evidence_block(records),
        "non_empty_gold": _evidence_block([r for r in records if r["evidence"]["gold_set_size"] > 0]),
        "empty_gold": _evidence_block([r for r in records if r["evidence"]["gold_set_size"] == 0]),
        "status_counts": dict(sorted(Counter(r["evidence"]["status"] for r in records).items())),
        "invalid_reference_cases": _ratio(
            sum(bool(r["evidence"]["invalid_reference_count"]) for r in records), len(records)
        ),
        "invalid_reference_count": sum(r["evidence"]["invalid_reference_count"] for r in records),
    }


def _native_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Diagnostic-only counts of the graph's raw verdicts. Not an evaluation result."""
    unclear = [r for r in records if _is_native_unclear(r)]
    return {
        "native_verdict": dict(sorted(Counter(
            r["native_verdict"] if r["native_verdict"] is not None else "MISSING" for r in records
        ).items())),
        "native_unclear_outputs": _ratio(len(unclear), len(records)),
        "native_unclear_successful": _ratio(
            sum(r["execution_status"] == "success" for r in unclear), len(records)
        ),
        "native_unclear_prediction_failures": _ratio(
            sum(r["execution_status"] != "success" for r in unclear), len(records)
        ),
    }


def _compliance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    outcomes = Counter(r["compliance"]["outcome_status"] for r in records)
    confusion = Counter(r["compliance"]["confusion_category"] for r in records
                        if r["compliance"]["confusion_category"])
    tp, tn, fp, fn = (confusion[name] for name in ("TP", "TN", "FP", "FN"))
    gold_fail = [r for r in records if r["compliance"]["gold_verdict"] == "FAIL"]
    breach = Counter(r["compliance"]["breach_outcome"] for r in gold_fail)
    return {
        "outcome_population": {
            "total": total,
            "binary_decisions": outcomes["binary_decision"],
            "abstentions": outcomes["abstention"],
            "prediction_failures": outcomes["prediction_failure"],
            "unexpected_not_applicable": outcomes["unexpected_not_applicable"],
        },
        "decision_coverage": _ratio(outcomes["binary_decision"], total),
        "abstention_rate": _ratio(outcomes["abstention"], total),
        "prediction_failure_rate": _ratio(outcomes["prediction_failure"], total),
        "unexpected_not_applicable_rate": _ratio(outcomes["unexpected_not_applicable"], total),
        "binary": {
            "decisions": tp + tn + fp + fn,
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
            "breach_precision": _ratio(tp, tp + fp),
            "breach_recall": _ratio(tp, tp + fn),
        },
        "breach_outcomes": {
            "gold_fail_total": len(gold_fail),
            "counts": {name: breach[name] for name in BREACH_OUTCOMES},
            "automatic_breach_detection_rate": _ratio(breach["detected"], len(gold_fail)),
            "false_clearance_rate": _ratio(breach["false_clearance"], len(gold_fail)),
            "abstention_rate": _ratio(breach["abstained"], len(gold_fail)),
            "prediction_failure_rate": _ratio(breach["prediction_failure"], len(gold_fail)),
            "unexpected_not_applicable_rate": _ratio(breach["unexpected_not_applicable"], len(gold_fail)),
        },
        "evidence": _evidence_summary(records),
        "native_diagnostics": _native_diagnostics(records),
    }


def _extraction_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    present = [r for r in records if r["extraction"]["gold_value_status"] == "PRESENT"]
    population = [r for r in records if r["extraction"]["evidence_coverage"] is not None]
    statuses = Counter(r["extraction"]["evidence_status"] for r in records)
    failures = [r for r in records if r["execution_status"] != "success"]
    return {
        "counts": {
            "total": total,
            "present_expected_values": len(present),
            "not_stated_expected_values": total - len(present),
            "unsupported": sum(r["extraction"]["evaluator_support"] == "UNSUPPORTED" for r in records),
            "coverage_population": len(population),
            "found": statuses["FOUND"],
            "not_found": statuses["NOT_FOUND"],
            "prediction_failures": len(failures),
            "coverage_population_prediction_failures": sum(r["execution_status"] != "success" for r in population),
        },
        "extraction_evidence_coverage": _ratio(
            sum(r["extraction"]["evidence_coverage"] for r in population), len(population)
        ),
        "prediction_failure_rate": _ratio(len(failures), total),
        "evidence": _evidence_summary(records),
        "native_diagnostics": _native_diagnostics(records),
    }


def _aggregate_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    compliance = [r for r in records if r["evaluation_family"] == "compliance"]
    extraction = [r for r in records if r["evaluation_family"] == "extraction"]
    unclear = [r for r in records if _is_native_unclear(r)]

    def count(family: str, success: bool) -> int:
        return sum(
            r["evaluation_family"] == family and (r["execution_status"] == "success") == success
            for r in unclear
        )

    return {
        "counts": {
            "total_questions": len(records),
            "compliance_questions": len(compliance),
            "extraction_questions": len(extraction),
            "execution_status": dict(sorted(Counter(r["execution_status"] for r in records).items())),
            "failure_category": dict(sorted(Counter(
                r["failure_category"] or "unspecified" for r in records
                if r["execution_status"] != "success").items())),
            "native_verdict": dict(sorted(Counter(
                r["native_verdict"] if r["native_verdict"] is not None else "MISSING" for r in records
            ).items())),
            "native_unclear_reconciliation": {
                "total_native_unclear": len(unclear),
                "compliance_abstentions": count("compliance", True),
                "compliance_prediction_failures": count("compliance", False),
                "extraction_successful_outputs": count("extraction", True),
                "extraction_prediction_failures": count("extraction", False),
            },
        },
        "compliance": _compliance_summary(compliance) if compliance else None,
        "extraction": _extraction_summary(extraction) if extraction else None,
        # Diagnostic only: mixes both families. Use the family-specific evidence blocks.
        "evidence_all_questions": _evidence_summary(records),
    }


def aggregate_scores(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate independently overall, by call ID, and by question ID."""
    call_ids = sorted({record["call_id"] for record in records})
    question_ids = sorted({record["question_id"] for record in records})
    return {
        "overall": _aggregate_group(records),
        "by_call_id": {
            call_id: _aggregate_group([record for record in records if record["call_id"] == call_id])
            for call_id in call_ids
        },
        "by_question_id": {
            question_id: _aggregate_group([record for record in records if record["question_id"] == question_id])
            for question_id in question_ids
        },
    }
