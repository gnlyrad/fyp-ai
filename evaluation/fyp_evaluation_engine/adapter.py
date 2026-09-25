"""Normalize the final graph state into stable call-question predictions."""
from __future__ import annotations

from typing import Any

from fyp_evaluation_loader import evaluation_family_for_question_type

VERDICT_MAP = {
    "Pass": "PASS",
    "Fail": "FAIL",
    "Unclear": "UNCLEAR",
    "N/A": "NOT_APPLICABLE",
}
FAILURE_EVIDENCE_STATUSES = {"analysis_unavailable", "classification_unavailable"}


def _family(item: dict[str, Any]) -> str:
    return item.get("evaluation_family") or evaluation_family_for_question_type(item["type"])


def _error(
    category: str, call_id: str, question_id: str | None, message: str,
    family: str | None = None, **details: Any,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "category": category,
        "call_id": call_id,
        "question_id": question_id,
        "evaluation_family": family,
        "message": message,
    }
    if details:
        value["details"] = details
    return value


def _base_prediction(call_id: str, item: dict[str, Any], latency_ms: float) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "question_id": item["id"],
        "question_type": item["type"],
        "evaluation_family": _family(item),
        "requirement_type": item.get("requirement_type"),
        "field": item.get("field"),
        "predicted_verdict": None,
        "native_verdict": None,
        "selected_evidence": [],
        "predicted_evidence_turn_ids": [],
        "unresolved_evidence_references": [],
        "evidence_reference_keys": [],
        "evidence_references_valid": None,
        "evidence_availability": "unavailable",
        "execution_status": "prediction_failure",
        "failure_category": "missing_prediction",
        "failure_message": None,
        "call_latency_ms": latency_ms,
        "graph_metadata": {},
    }


def _findings(graph_output: Any) -> tuple[list[Any] | None, str | None]:
    if not isinstance(graph_output, dict):
        return None, "Graph output is not an object."
    report = graph_output.get("report")
    if not isinstance(report, dict):
        return None, "Graph output does not contain a final report object."
    findings = report.get("findings")
    if not isinstance(findings, list):
        return None, "Final report findings are missing or malformed."
    return findings, None


def _chunk_map(graph_output: Any) -> tuple[dict[str, dict[str, Any]], set[str]]:
    if not isinstance(graph_output, dict) or not isinstance(graph_output.get("chunks"), list):
        return {}, set()
    result: dict[str, dict[str, Any]] = {}
    duplicates: set[str] = set()
    for chunk in graph_output["chunks"]:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("chunk_id"), str):
            continue
        chunk_id = chunk["chunk_id"]
        if chunk_id in result:
            duplicates.add(chunk_id)
        else:
            result[chunk_id] = chunk
    return result, duplicates


def _map_evidence(
    call_id: str,
    question_id: str,
    family: str,
    finding: dict[str, Any],
    chunks: dict[str, dict[str, Any]],
    duplicate_chunk_ids: set[str],
    transcript_turn_ids: set[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    spans = finding.get("spans")
    if not isinstance(spans, list):
        return {
            "selected_evidence": [],
            "predicted_evidence_turn_ids": [],
            "unresolved_evidence_references": [{"reason": "finding.spans is not a list"}],
            "evidence_reference_keys": ["invalid:malformed_spans"],
            "evidence_references_valid": False,
            "evidence_availability": "malformed",
        }, [_error("malformed_evidence", call_id, question_id, "finding.spans is not a list", family)]

    turn_ids: set[Any] = set()
    reference_keys: set[str] = set()
    unresolved: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for index, span in enumerate(spans):
        if not isinstance(span, dict):
            ref = {"span_index": index, "reason": "span is not an object"}
            unresolved.append(ref)
            reference_keys.add(f"invalid:span:{index}")
            continue
        selected.append(dict(span))
        chunk_id = span.get("chunk_id")
        reason = None
        source_segment_id = None
        if not isinstance(chunk_id, str) or not chunk_id:
            reason = "span has no chunk_id"
        elif chunk_id in duplicate_chunk_ids:
            reason = "chunk_id is duplicated in graph output"
        elif chunk_id not in chunks:
            reason = "chunk_id does not resolve to a graph chunk"
        else:
            # Deliberately do not infer a turn by parsing a name such as chunk-04.
            source_segment_id = chunks[chunk_id].get("source_segment_id")
            if source_segment_id is None:
                reason = "resolved chunk has no source_segment_id"
            elif source_segment_id not in transcript_turn_ids:
                reason = "source_segment_id does not resolve to a transcript turn"
        if reason:
            unresolved.append({"span_index": index, "chunk_id": chunk_id, "reason": reason})
            reference_keys.add(f"invalid:{chunk_id or 'missing'}:{index}")
        else:
            turn_ids.add(source_segment_id)
            reference_keys.add(f"turn:{source_segment_id}")

    if unresolved:
        errors.append(_error(
            "invalid_evidence_reference",
            call_id,
            question_id,
            "One or more final evidence spans could not be mapped to transcript turns.",
            family,
            references=unresolved,
        ))
    return {
        "selected_evidence": selected,
        "predicted_evidence_turn_ids": sorted(turn_ids, key=str),
        "unresolved_evidence_references": unresolved,
        "evidence_reference_keys": sorted(reference_keys),
        "evidence_references_valid": not unresolved,
        "evidence_availability": "invalid_references" if unresolved else "available",
    }, errors


def _fail(prediction: dict[str, Any], category: str, message: str) -> None:
    """Mark one question-level prediction as unusable; keep the native verdict for audit."""
    prediction["execution_status"] = "prediction_failure"
    prediction["failure_category"] = category
    prediction["failure_message"] = message
    # A fallback verdict from a failed analysis is diagnostic only, never a decision.
    prediction["predicted_verdict"] = None


def normalize_call_output(
    call_id: str,
    checklist: list[dict[str, Any]],
    transcript_turn_ids: set[Any],
    graph_output: Any,
    latency_ms: float,
    call_error: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return one prediction for every expected question and diagnostics.

    Evidence comes only from each final finding's ``spans``. Candidate retrieval
    chunks and ``reviewed_chunk_ids`` are intentionally ignored.

    ``execution_status`` is ``success`` or ``prediction_failure``; the specific
    reason is ``failure_category``. ``native_verdict`` is the raw graph value.
    ``predicted_verdict`` is the normalized *compliance* verdict of a successful
    compliance question and is null otherwise.
    """
    errors: list[dict[str, Any]] = []
    predictions = {item["id"]: _base_prediction(call_id, item, latency_ms) for item in checklist}

    def missing(message: str) -> None:
        for question_id, prediction in predictions.items():
            prediction["failure_message"] = message
            errors.append(_error(
                "missing_prediction", call_id, question_id, message, prediction["evaluation_family"],
                failure_category=prediction["failure_category"], native_verdict=None,
            ))

    if call_error is not None:
        errors.append(_error("graph_exception", call_id, None, call_error))
        missing("Graph execution failed before a prediction was available.")
        return list(predictions.values()), errors

    findings, findings_error = _findings(graph_output)
    if findings_error:
        errors.append(_error("malformed_output", call_id, None, findings_error))
        missing("No final finding was available.")
        return list(predictions.values()), errors

    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for index, finding in enumerate(findings or []):
        if not isinstance(finding, dict) or not isinstance(finding.get("checklist_id"), str):
            errors.append(_error("malformed_finding", call_id, None, "A finding has no string checklist_id.", finding_index=index))
            continue
        question_id = finding["checklist_id"]
        if question_id not in predictions:
            errors.append(_error("unexpected_prediction", call_id, question_id, "Graph returned a finding not present in the published checklist."))
            continue
        if question_id in by_id:
            duplicate_ids.add(question_id)
        else:
            by_id[question_id] = finding

    chunks, duplicate_chunk_ids = _chunk_map(graph_output)
    report = graph_output.get("report", {})
    call_metadata = {
        "report_summary": report.get("summary"),
        "consistency": report.get("consistency"),
        "warnings": report.get("warnings", []),
    }
    for key in ("usage_metadata", "response_metadata", "trace_id"):
        if key in graph_output:
            call_metadata[key] = graph_output[key]

    for item in checklist:
        question_id = item["id"]
        prediction = predictions[question_id]
        family = prediction["evaluation_family"]
        finding = by_id.get(question_id)

        def fail(category: str, message: str, native: Any = None, **details: Any) -> None:
            _fail(prediction, category, message)
            errors.append(_error(
                category if category in {"duplicate_prediction", "missing_prediction", "malformed_verdict"}
                else "prediction_failure",
                call_id, question_id, message, family,
                failure_category=category, native_verdict=native, **details,
            ))

        if question_id in duplicate_ids:
            fail("duplicate_prediction", "Graph returned more than one final finding for this question.")
            continue
        if finding is None:
            fail("missing_prediction", "Graph returned no final finding for this question.")
            continue

        native_verdict = finding.get("verdict")
        prediction["native_verdict"] = native_verdict
        evidence, evidence_errors = _map_evidence(
            call_id, question_id, family, finding, chunks, duplicate_chunk_ids, transcript_turn_ids
        )
        prediction.update(evidence)
        errors.extend(evidence_errors)
        finding_metadata = {
            key: finding.get(key)
            for key in (
                "question_type", "reason", "pass_condition", "evidence_status",
                "analysis_coverage", "applicability_coverage", "classification_status",
                "classification_error", "requires_review", "applicability",
                "retrieval_diagnostics",
            )
            if key in finding
        }
        prediction["graph_metadata"] = {**call_metadata, "finding": finding_metadata}
        if native_verdict not in VERDICT_MAP:
            fail("malformed_verdict", f"Unsupported native verdict: {native_verdict!r}", native_verdict)
        elif finding.get("evidence_status") in FAILURE_EVIDENCE_STATUSES:
            # The graph may catch an LLM exception and emit Unclear. Keep the
            # verdict for audit, but do not count that operational failure as an abstention.
            fail(
                finding["evidence_status"],
                f"Graph reported {finding.get('evidence_status')} rather than a completed judgement.",
                native_verdict,
                evidence_status=finding.get("evidence_status"),
                classification_error=finding.get("classification_error"),
                graph_reason=finding.get("reason"),
            )
        elif prediction["evidence_availability"] == "malformed":
            fail("malformed_evidence", "Final finding evidence spans are malformed.", native_verdict)
        else:
            prediction["execution_status"] = "success"
            prediction["failure_category"] = None
            prediction["failure_message"] = None
            prediction["predicted_verdict"] = VERDICT_MAP[native_verdict] if family == "compliance" else None

    return list(predictions.values()), errors
