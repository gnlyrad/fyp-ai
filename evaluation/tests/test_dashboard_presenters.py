"""Presentation logic: typed rows, drill populations, run status and evidence wording."""
import json
from pathlib import Path

import pytest

from dashboard.domain import RunRecord
from dashboard.presenters import (
    DRILLS,
    drill_count,
    drills,
    evidence_explanation,
    load_presentation,
    matches,
    problems,
    result_outcome,
    result_row,
    run_status,
)
from fyp_evaluation_engine.scoring import aggregate_scores
from helpers import extraction_gold, extraction_prediction, score_compliance, score_extraction
from fyp_evaluation_engine.scoring import score_prediction

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
PRESENTATION = load_presentation(EVALUATION_ROOT / "metric_presentation.json")


def _scores():
    """Every compliance outcome and extraction status, spread over two calls."""
    records = []
    for call in ("CALL_001", "CALL_002"):
        cases = [
            ("C01", "FAIL", "FAIL", {}), ("C02", "FAIL", "PASS", {}), ("C03", "PASS", "FAIL", {}),
            ("C04", "PASS", "PASS", {}), ("C05", "FAIL", "UNCLEAR", {}), ("C06", "PASS", "UNCLEAR", {}),
            ("C07", "FAIL", "NOT_APPLICABLE", {}),
            ("C08", "FAIL", None, {"status": "prediction_failure", "failure_category": "analysis_unavailable"}),
            ("C09", "PASS", None, {"status": "prediction_failure", "failure_category": "missing_prediction"}),
        ]
        for question_id, gold, verdict, extra in cases:
            records.append(score_compliance(gold, verdict, call_id=call, question_id=question_id, **extra))
        evidence = [{"chunk_id": "c", "text": "My NRIC is S1234567A"}]
        records += [
            score_extraction("S1234567A", call_id=call, question_id="K01", evidence=evidence, turns=[1]),
            score_extraction("S7654321Z", call_id=call, question_id="K02", evidence=evidence, turns=[1]),
            score_extraction("SGD", call_id=call, question_id="K03", status="prediction_failure",
                             failure_category="missing_prediction"),
            score_extraction("1 Jan 2000", "unknown_type", call_id=call, question_id="K04"),
            score_prediction(extraction_prediction(call_id=call, question_id="K05"), extraction_gold(None)),
            score_prediction(extraction_prediction(call_id=call, question_id="K06", status="prediction_failure"),
                             extraction_gold(None)),
        ]
    return records


def _nodes(summary):
    yield "overall", summary["overall"]
    for key in ("by_call_id", "by_question_id"):
        yield from ((f"{key}.{name}", node) for name, node in summary[key].items())


def test_every_drill_filter_reproduces_its_summary_count_at_every_level():
    scores = _scores()
    summary = aggregate_scores(scores)
    rows = [result_row(score) for score in scores]
    for name, node in _nodes(summary):
        node_rows = [row for row in rows if f"by_call_id.{row['call_id']}" == name
                     or f"by_question_id.{row['question_id']}" == name or name == "overall"]
        for key, (_, row_filter) in DRILLS.items():
            family = row_filter["family"]
            if node.get(family) is None:
                continue
            assert sum(matches(row, row_filter) for row in node_rows) == drill_count(node, key), (name, key)


def test_drill_registry_is_labelled_for_the_frontend():
    served = drills(PRESENTATION)
    assert set(served) == set(DRILLS)
    assert all(item["label"] and item["filter"]["family"] in {"compliance", "extraction"} for item in served.values())


@pytest.mark.parametrize("run_dir", sorted((EVALUATION_ROOT / "runs").glob("*/scores.jsonl")))
def test_drills_reproduce_counts_on_saved_runs(run_dir):
    scores = [json.loads(line) for line in run_dir.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    summary = json.loads((run_dir.parent / "summary.json").read_text(encoding="utf-8-sig"))
    rows = [result_row(score) for score in scores]
    for key, (_, row_filter) in DRILLS.items():
        if summary["overall"].get(row_filter["family"]) is not None:
            assert sum(matches(row, row_filter) for row in rows) == drill_count(summary["overall"], key), key


def test_result_outcome_labels_use_presentation_wording():
    def outcome(score):
        return result_outcome(result_row(score), PRESENTATION)

    assert outcome(score_compliance("FAIL", "PASS")) == {"label": "Missed breach", "tone": "bad"}
    assert outcome(score_compliance("PASS", "PASS")) == {"label": "Correct pass", "tone": "good"}
    assert outcome(score_compliance("FAIL", "UNCLEAR"))["label"] == "UNCLEAR"
    assert outcome(score_compliance("FAIL", None, status="prediction_failure"))["label"] == "Prediction failed"
    assert outcome(score_extraction("SGD", evidence=[{"text": "in SGD"}]))["label"] == "Successfully located"
    assert outcome(score_extraction("SGD", evidence=[{"text": "in USD"}]))["label"] == "Expected value not located"
    no_value = score_prediction(extraction_prediction(), extraction_gold(None))
    assert outcome(no_value) == {"label": "No value expected", "tone": "neutral"}


def _run(status, failures=0):
    counts = {"success": 5, **({"prediction_failure": failures} if failures else {})}
    return RunRecord(run_id="r", dataset_id="d", source="Local", metadata={"execution_status": status},
                     summary={"overall": {"counts": {"execution_status": counts}}})


def test_run_status_separates_warnings_from_prediction_failures():
    warnings_only = run_status(_run("completed_with_errors"), [{"category": "pipeline_diagnostic"}], PRESENTATION)
    assert warnings_only["label"] == "Completed with issues"
    assert warnings_only["detail"] == "1 pipeline warning"
    assert warnings_only["prediction_failures"] == 0
    both = run_status(_run("completed_with_errors", 2),
                      [{"category": "pipeline_diagnostic"}] * 3 + [{"category": "prediction_failure"}] * 2, PRESENTATION)
    assert both["detail"] == "2 prediction failures · 3 pipeline warnings"
    assert run_status(_run("completed"), [], PRESENTATION) | {"detail": ""} == run_status(_run("completed"), [], PRESENTATION)
    assert run_status(_run("completed"), [], PRESENTATION)["label"] == "Complete"
    assert run_status(_run("completed_with_errors"), None, PRESENTATION)["pipeline_warnings"] is None


def test_problems_list_only_non_zero_issues_with_their_drill():
    scores = _scores()
    summary = aggregate_scores(scores)["overall"]
    errors = [{"category": "pipeline_diagnostic"}, {"category": "prediction_failure"}]
    found = problems(summary, errors, PRESENTATION)
    drill_keys = [item["drill"] for item in found if item["target"] == "cases"]
    assert drill_keys == ["confusion.FN", "compliance.prediction_failure", "confusion.FP", "compliance.abstention",
                          "compliance.unexpected_not_applicable", "extraction.not_found",
                          "extraction.prediction_failure", "extraction.unsupported"]
    # Question-level failure errors are counted from the summary, not repeated from errors.jsonl.
    assert [item["category"] for item in found if item["target"] == "diagnostics"] == ["pipeline_diagnostic"]
    clean = aggregate_scores([score_compliance("PASS", "PASS")])["overall"]
    assert problems(clean, [], PRESENTATION) == []


def _row(gold, predicted, status="scored", invalid=0):
    return {"gold_turns": gold, "predicted_turns": predicted, "evidence_status": status,
            "invalid_reference_count": invalid}


@pytest.mark.parametrize(("row", "sentence"), [
    (_row([3, 4], [4]), "The model selected 1 of 2 expected turns, missed turn 3."),
    (_row([3, 4], [4, 5]), "The model selected 1 of 2 expected turns, missed turn 3 and added turn 5, which was not expected."),
    (_row([3], [3]), "The model selected exactly the expected evidence (turn 3)."),
    (_row([], []), "No evidence was expected, and the model selected none."),
    (_row([], [6]), "No evidence was expected, but the model selected turn 6."),
    (_row([2, 7], []), "The model selected no evidence and missed the expected turns 2, 7."),
    (_row([3], [3], invalid=1),
     "The model selected 1 of 1 expected turn. 1 selected reference could not be matched to a transcript turn."),
    (_row([3], [], status="prediction_failure"), "No evidence was scored because the prediction failed."),
])
def test_evidence_explanation_is_derived_from_the_sets(row, sentence):
    assert evidence_explanation(row) == sentence
