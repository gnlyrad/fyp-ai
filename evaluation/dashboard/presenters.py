"""Pure presentation logic for the dashboard API.

Everything here reads values the evaluator already produced (summary.json,
scores.jsonl, errors.jsonl, metadata.json) and reshapes or words them for
people. Nothing here computes a metric: drill-down populations are filters
over score records whose sizes must equal the summary counts.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .domain import RunRecord
from .view_models import metric_index, metric_value, nested

PREDICTION_FAILURE = "prediction_failure"
PIPELINE_DIAGNOSTIC = "pipeline_diagnostic"
# errors.jsonl categories with no score-record counterpart; the rest duplicate
# prediction failures that "Where to look" already counts from the summary.
DIAGNOSTIC_ONLY_CATEGORIES = frozenset({
    PIPELINE_DIAGNOSTIC, "invalid_evidence_reference", "graph_exception",
    "malformed_output", "malformed_finding", "unexpected_prediction",
})

# Columns of the run-history table, in order.
RUN_HISTORY_METRICS = (
    "compliance.decision_coverage",
    "compliance.automatic_breach_detection_rate",
    "compliance.false_clearance_rate",
    "compliance.prediction_failure_rate",
    "extraction.evidence_coverage",
)

# Metrics compared between two runs (same set the Streamlit comparison used).
COMPARISON_METRICS = (
    "compliance.decision_coverage",
    "compliance.abstention_rate",
    "compliance.prediction_failure_rate",
    "compliance.breach_recall",
    "compliance.breach_precision",
    "compliance.binary_accuracy",
    "compliance.automatic_breach_detection_rate",
    "compliance.false_clearance_rate",
    "compliance.gold_breach_abstention_rate",
    "compliance.gold_breach_prediction_failure_rate",
    "extraction.evidence_coverage",
    "extraction.prediction_failure_rate",
)


def load_presentation(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


# --------------------------------------------------------------------------- rows

def _ratio_value(ratio: Any) -> float | None:
    return ratio.get("value") if isinstance(ratio, dict) else None


def result_row(score: dict[str, Any], question: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten one typed score record into the row every drill-down filters on."""
    compliance = score.get("compliance") or {}
    extraction = score.get("extraction") or {}
    evidence = score.get("evidence") or {}
    question = question or {}
    return {
        "call_id": score["call_id"],
        "question_id": score["question_id"],
        "family": score.get("evaluation_family"),
        "question": question.get("question"),
        "field": score.get("field"),
        "requirement_type": score.get("requirement_type"),
        "execution_status": score.get("execution_status"),
        "failure_category": score.get("failure_category"),
        "native_verdict": score.get("native_verdict"),
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
        "gold_turns": list(score.get("gold_evidence_turn_ids") or []),
        "predicted_turns": list(score.get("predicted_evidence_turn_ids") or []),
        "evidence_status": evidence.get("status"),
        "evidence_precision": _ratio_value(evidence.get("precision")),
        "evidence_recall": _ratio_value(evidence.get("recall")),
        "evidence_exact_match": _ratio_value(evidence.get("exact_set_match")),
        "invalid_reference_count": evidence.get("invalid_reference_count") or 0,
    }


def matches(row: dict[str, Any], row_filter: dict[str, Any]) -> bool:
    """Same semantics as the frontend filter: every key equals its value."""
    return all(row.get(key) == value for key, value in row_filter.items())


def result_outcome(row: dict[str, Any], presentation: dict[str, Any]) -> dict[str, str]:
    """One human label and a neutral tone for a row's result, from existing fields only."""
    statuses = presentation["statuses"]
    if row["execution_status"] == PREDICTION_FAILURE:
        return {"label": statuses["execution_status"][PREDICTION_FAILURE]["label"], "tone": "bad"}
    if row["family"] == "compliance":
        outcome = row["compliance_outcome_status"]
        if outcome == "binary_decision":
            category = row["confusion_category"]
            label = presentation["confusion_categories"][category]["label"]
            return {"label": label, "tone": "good" if category in {"TP", "TN"} else "bad"}
        return {"label": statuses["compliance_outcome_status"][outcome]["label"], "tone": "warn"}
    if row["gold_value_status"] == "NOT_STATED":
        return {"label": statuses["extraction_gold_value_status"]["NOT_STATED"]["label"], "tone": "neutral"}
    if row["evaluator_support"] == "UNSUPPORTED":
        return {"label": statuses["evaluator_support"]["UNSUPPORTED"]["label"], "tone": "neutral"}
    status = row["extraction_evidence_status"]
    return {
        "label": statuses["extraction_evidence_status"][status]["label"],
        "tone": "good" if status == "FOUND" else "bad",
    }


# --------------------------------------------------------------------------- run status

def run_status(run: RunRecord, errors: list[dict[str, Any]] | None, presentation: dict[str, Any]) -> dict[str, Any]:
    """Human run status. Counts separate prediction failures from pipeline warnings,
    because the persisted status is completed_with_errors for either."""
    raw = run.status
    entry = presentation["run_status"].get(raw, {"label": raw.replace("_", " ").capitalize(), "description": ""})
    failures = int(nested(run.summary, "overall.counts.execution_status.prediction_failure", 0) or 0)
    categories = Counter(error.get("category") for error in errors or [])
    warnings = categories.get(PIPELINE_DIAGNOSTIC, 0) if errors is not None else None
    parts = []
    if failures:
        parts.append(f"{failures} prediction failure{'s' if failures != 1 else ''}")
    if warnings:
        parts.append(f"{warnings} pipeline warning{'s' if warnings != 1 else ''}")
    tone = "bad" if raw == "failed" else "warn" if raw == "completed_with_errors" else "good"
    return {
        "raw": raw,
        "label": entry["label"],
        "description": entry["description"],
        "tone": tone,
        "prediction_failures": failures,
        "pipeline_warnings": warnings,
        "detail": " · ".join(parts),
    }


# --------------------------------------------------------------------------- drills

_C, _E = {"family": "compliance"}, {"family": "extraction"}

# Every drill-down population: the summary count it reproduces (relative to any
# aggregate node) and the row filter that selects exactly those results.
DRILLS: dict[str, tuple[str, dict[str, Any]]] = {
    "compliance.all": ("compliance.outcome_population.total", _C),
    "compliance.binary_decision": ("compliance.outcome_population.binary_decisions",
                                   {**_C, "compliance_outcome_status": "binary_decision"}),
    "compliance.abstention": ("compliance.outcome_population.abstentions",
                              {**_C, "compliance_outcome_status": "abstention"}),
    "compliance.prediction_failure": ("compliance.outcome_population.prediction_failures",
                                      {**_C, "compliance_outcome_status": PREDICTION_FAILURE}),
    "compliance.unexpected_not_applicable": ("compliance.outcome_population.unexpected_not_applicable",
                                             {**_C, "compliance_outcome_status": "unexpected_not_applicable"}),
    **{f"confusion.{name}": (f"compliance.binary.confusion_matrix.{name}", {**_C, "confusion_category": name})
       for name in ("TP", "TN", "FP", "FN")},
    "breach.all": ("compliance.breach_outcomes.gold_fail_total", {**_C, "gold_verdict": "FAIL"}),
    **{f"breach.{name}": (f"compliance.breach_outcomes.counts.{name}", {**_C, "breach_outcome": name})
       for name in ("detected", "false_clearance", "abstained", PREDICTION_FAILURE, "unexpected_not_applicable")},
    "extraction.all": ("extraction.counts.total", _E),
    "extraction.present": ("extraction.counts.present_expected_values", {**_E, "gold_value_status": "PRESENT"}),
    "extraction.not_stated": ("extraction.counts.not_stated_expected_values", {**_E, "gold_value_status": "NOT_STATED"}),
    "extraction.coverage_population": ("extraction.counts.coverage_population", {**_E, "evaluator_support": "SUPPORTED"}),
    "extraction.found": ("extraction.counts.found", {**_E, "extraction_evidence_status": "FOUND"}),
    "extraction.not_found": ("extraction.counts.not_found", {**_E, "extraction_evidence_status": "NOT_FOUND"}),
    "extraction.coverage_failure": ("extraction.counts.coverage_population_prediction_failures",
                                    {**_E, "evaluator_support": "SUPPORTED", "execution_status": PREDICTION_FAILURE}),
    "extraction.prediction_failure": ("extraction.counts.prediction_failures",
                                      {**_E, "execution_status": PREDICTION_FAILURE}),
    "extraction.unsupported": ("extraction.counts.unsupported", {**_E, "evaluator_support": "UNSUPPORTED"}),
}


def drill_labels(presentation: dict[str, Any]) -> dict[str, str]:
    statuses, confusion = presentation["statuses"], presentation["confusion_categories"]
    failed = statuses["execution_status"][PREDICTION_FAILURE]["label"]
    labels = {
        "compliance.all": "All compliance checks",
        **{f"compliance.{key}": value["label"] for key, value in statuses["compliance_outcome_status"].items()},
        **{f"confusion.{key}": value["label"] for key, value in confusion.items()},
        "breach.all": "All actual breaches",
        **{f"breach.{key}": value["label"] for key, value in statuses["breach_outcome"].items()},
        "extraction.all": "All extraction checks",
        "extraction.present": statuses["extraction_gold_value_status"]["PRESENT"]["label"],
        "extraction.not_stated": statuses["extraction_gold_value_status"]["NOT_STATED"]["label"],
        "extraction.coverage_population": "Expected values the evaluator can check",
        "extraction.found": statuses["extraction_evidence_status"]["FOUND"]["label"],
        "extraction.not_found": statuses["extraction_evidence_status"]["NOT_FOUND"]["label"],
        "extraction.coverage_failure": failed,
        "extraction.prediction_failure": failed,
        "extraction.unsupported": statuses["evaluator_support"]["UNSUPPORTED"]["label"],
    }
    return {key: labels[key] for key in DRILLS}


def drills(presentation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    labels = drill_labels(presentation)
    return {key: {"label": labels[key], "count_path": path, "filter": row_filter}
            for key, (path, row_filter) in DRILLS.items()}


def drill_count(node: dict[str, Any], key: str) -> int:
    return int(nested(node, DRILLS[key][0], 0) or 0)


# "Where to look", in reading order. Only non-zero entries are shown.
PROBLEM_DRILLS = (
    "confusion.FN",
    "compliance.prediction_failure",
    "confusion.FP",
    "compliance.abstention",
    "compliance.unexpected_not_applicable",
    "extraction.not_found",
    "extraction.prediction_failure",
    "extraction.unsupported",
)


def problems(summary_node: dict[str, Any], errors: list[dict[str, Any]], presentation: dict[str, Any]) -> list[dict[str, Any]]:
    """Non-zero issue counts for "Where to look", each linked to its drill or diagnostics category."""
    labels = drill_labels(presentation)
    found = []
    for key in PROBLEM_DRILLS:
        count = drill_count(summary_node, key)
        if count:
            found.append({"family": DRILLS[key][1]["family"], "label": labels[key], "count": count,
                          "drill": key, "target": "cases"})
    categories = presentation["error_categories"]
    for category, count in sorted(Counter(error.get("category") for error in errors).items(), key=lambda item: str(item[0])):
        if category in DIAGNOSTIC_ONLY_CATEGORIES:
            found.append({"family": "pipeline", "label": categories.get(category, {}).get("label", category),
                          "count": count, "category": category, "target": "diagnostics"})
    return found


# --------------------------------------------------------------------------- run history / compare

def ratio(summary_node: dict[str, Any], definition: dict[str, Any]) -> dict[str, Any] | None:
    value = metric_value(summary_node, definition)
    if not isinstance(value, dict) or "denominator" not in value:
        return None
    return {key: value.get(key) for key in ("value", "percentage", "numerator", "denominator")}


# Operational comparison rows: id, unit, and how to read each from `run_operations`.
OPERATION_IDS = ("latency_total", "latency_per_call", "tokens_total", "cost_total", "cost_per_call")
OPERATION_UNITS = {
    "latency_total": "ms", "latency_per_call": "ms", "tokens_total": "tokens",
    "cost_total": "currency", "cost_per_call": "currency",
}


def _number_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def run_operations(run: RunRecord) -> dict[str, Any]:
    """Time, tokens and estimated cost of a run, from metadata only.

    Fields are None when the run did not record them (older runs), never zero.
    """
    metadata = run.metadata
    executions = metadata.get("call_executions") or []
    latencies = [item["latency_ms"] for item in executions if _number_or_none(item.get("latency_ms")) is not None]
    calls = len(executions)
    observability = metadata.get("observability")
    observability = observability if isinstance(observability, dict) else {}
    usage = observability.get("token_usage")
    usage = usage if isinstance(usage, dict) else None
    cost = observability.get("estimated_cost")
    cost = cost if isinstance(cost, dict) else None
    amount = _number_or_none(cost.get("amount")) if cost else None
    return {
        "calls": calls,
        "total_latency_ms": _number_or_none(metadata.get("total_latency_ms")),
        "avg_call_latency_ms": sum(latencies) / len(latencies) if latencies else None,
        "tokens": ({key: usage.get(key) for key in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens")}
                   if usage else None),
        "cost": ({"amount": amount, "currency": cost.get("currency", "USD"),
                  "per_call": amount / calls if amount is not None and calls else None,
                  "unpriced_models": cost.get("unpriced_models") or [],
                  "prices": cost.get("prices_used") if isinstance(cost.get("prices_used"), dict) else {}} if cost else None),
    }


def operation_values(operations: dict[str, Any]) -> dict[str, float | None]:
    cost, tokens = operations["cost"], operations["tokens"]
    return {
        "latency_total": operations["total_latency_ms"],
        "latency_per_call": operations["avg_call_latency_ms"],
        "tokens_total": _number_or_none(tokens["total_tokens"]) if tokens else None,
        "cost_total": cost["amount"] if cost else None,
        "cost_per_call": cost["per_call"] if cost else None,
    }


def run_summary(run: RunRecord, definitions: dict[str, Any], presentation: dict[str, Any]) -> dict[str, Any]:
    index = metric_index(definitions)
    overall = run.summary.get("overall", {})
    return {
        "run_id": run.run_id,
        "dataset_id": run.dataset_id,
        "evaluated_at": run.evaluated_at,
        "is_complete": run.is_complete,
        "status": run_status(run, None, presentation),
        "graph_commit": run.metadata.get("graph_commit"),
        "graph_worktree_dirty": run.metadata.get("graph_worktree_dirty") is True,
        "chat_model_id": nested(run.metadata, "model_configuration.chat_model_id"),
        "run_kind": run.metadata.get("run_kind"),
        "metrics": {metric_id: ratio(overall, index[metric_id]) for metric_id in RUN_HISTORY_METRICS},
        "operations": run_operations(run),
    }


def lacks_run_fingerprints(run: RunRecord) -> bool:
    """True for runs saved before checklist and metric-definition fingerprints were recorded."""
    return not (run.metadata.get("metric_definitions_sha256")
                and nested(run.metadata, "dataset_snapshot.checklist_sha256"))


def compare_runs(first: RunRecord, second: RunRecord, definitions: dict[str, Any], incompatibilities: list[str]) -> dict[str, Any]:
    """Compare two runs. The earlier run (by evaluation time) is always `earlier`, so every
    difference reads 'later minus earlier' regardless of the order the caller passed."""
    earlier, later = sorted((first, second), key=lambda run: run.evaluated_at or "")
    index = metric_index(definitions)
    rows = []
    for metric_id in COMPARISON_METRICS:
        left = ratio(earlier.summary.get("overall", {}), index[metric_id])
        right = ratio(later.summary.get("overall", {}), index[metric_id])
        left_pct = left and left["percentage"]
        right_pct = right and right["percentage"]
        rows.append({
            "metric_id": metric_id,
            "earlier": left,
            "later": right,
            "delta_pp": None if left_pct is None or right_pct is None else round(right_pct - left_pct, 2),
        })
    before, after = operation_values(run_operations(earlier)), operation_values(run_operations(later))
    operations = [{
        "id": operation_id,
        "unit": OPERATION_UNITS[operation_id],
        "earlier": before[operation_id],
        "later": after[operation_id],
        "delta": None if before[operation_id] is None or after[operation_id] is None
        else after[operation_id] - before[operation_id],
    } for operation_id in OPERATION_IDS]
    return {
        "comparable": not incompatibilities,
        "incompatibilities": incompatibilities,
        "unverifiable_run_ids": [run.run_id for run in (earlier, later) if lacks_run_fingerprints(run)],
        "earlier_run_id": earlier.run_id,
        "later_run_id": later.run_id,
        "metrics": rows,
        "operations": operations,
    }


# --------------------------------------------------------------------------- individual result

def _turns(values: list[int]) -> str:
    values = sorted(values)
    label = "turn" if len(values) == 1 else "turns"
    return f"{label} {', '.join(str(value) for value in values)}"


def evidence_explanation(row: dict[str, Any]) -> str:
    """Deterministic sentence describing the evidence sets, derived from the score only."""
    status = row["evidence_status"]
    if status == "prediction_failure":
        return "No evidence was scored because the prediction failed."
    if status == "evidence_failure":
        return "The result had no usable evidence, so evidence scored zero."
    gold, predicted = set(row["gold_turns"]), set(row["predicted_turns"])
    invalid = row["invalid_reference_count"]
    suffix = ""
    if invalid:
        suffix = (f" {invalid} selected reference{'s' if invalid != 1 else ''} could not be matched "
                  "to a transcript turn.")
    if not gold and not predicted:
        return "No evidence was expected, and the model selected none." + suffix
    if not gold:
        return f"No evidence was expected, but the model selected {_turns(list(predicted))}." + suffix
    if not predicted:
        return f"The model selected no evidence and missed the expected {_turns(list(gold))}." + suffix
    if gold == predicted and not invalid:
        return f"The model selected exactly the expected evidence ({_turns(list(gold))})."
    matched, missed, extra = gold & predicted, gold - predicted, predicted - gold
    sentence = f"The model selected {len(matched)} of {len(gold)} expected turn{'s' if len(gold) != 1 else ''}"
    clauses = []
    if missed:
        clauses.append(f"missed {_turns(list(missed))}")
    if extra:
        clauses.append(f"added {_turns(list(extra))}, which {'was' if len(extra) == 1 else 'were'} not expected")
    if clauses:
        sentence += ", " + " and ".join(clauses)
    return sentence + "." + suffix


def transcript_turns(transcript: dict[str, Any], speaker_roles: dict[str, str], row: dict[str, Any]) -> list[dict[str, Any]]:
    gold, predicted = set(row["gold_turns"]), set(row["predicted_turns"])
    turns = []
    for segment in transcript.get("audio_segments", []):
        turn_id = segment.get("id")
        speaker = str(segment.get("speaker_label"))
        turns.append({
            "id": turn_id,
            "speaker": speaker_roles.get(speaker, speaker),
            "text": segment.get("transcript") or "",
            "start_time": segment.get("start_time"),
            "expected": turn_id in gold,
            "selected": turn_id in predicted,
        })
    return turns
