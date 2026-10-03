"""metric_presentation.json may only word what metric_definitions.json defines."""
import json
import re
from pathlib import Path

from fyp_evaluation_engine.runner import run_status
from fyp_evaluation_engine.scoring import aggregate_scores
from helpers import score_compliance

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
DEFINITIONS = json.loads((EVALUATION_ROOT / "metric_definitions.json").read_text(encoding="utf-8-sig"))
PRESENTATION = json.loads((EVALUATION_ROOT / "metric_presentation.json").read_text(encoding="utf-8-sig"))

# Evaluator vocabulary that must not lead a sponsor-facing explanation.
TECHNICAL_TERMS = re.compile(
    r"\b(gold|native|graph|PRESENT|NOT_STATED|NOT_FOUND|SUPPORTED|UNSUPPORTED|normali[sz]ed|spans?|TP|TN|FP|FN)\b"
)


def test_every_metric_is_worded_and_no_unknown_metric_is():
    defined = {metric["id"] for metric in DEFINITIONS["metrics"]}
    assert set(PRESENTATION["metrics"]) == defined


def test_metric_entries_have_the_tooltip_fields():
    audiences = set(PRESENTATION["audiences"])
    roles = {metric["id"]: metric["role"] for metric in DEFINITIONS["metrics"]}
    for metric_id, entry in PRESENTATION["metrics"].items():
        assert entry["audience"] in audiences, metric_id
        assert entry["label"] and entry["what"] and entry["calculation"], metric_id
        assert set(entry) <= {"label", "audience", "what", "calculation", "important"}, metric_id
        if roles[metric_id] == "diagnostic":
            assert entry["audience"] == "developer", metric_id


def test_sponsor_wording_avoids_evaluator_vocabulary():
    for metric_id, entry in PRESENTATION["metrics"].items():
        if entry["audience"] != "summary":
            continue
        for key in ("label", "what", "calculation", "important"):
            assert not TECHNICAL_TERMS.search(entry.get(key, "")), (metric_id, key)


def test_status_wording_refers_only_to_defined_status_values():
    defined = {status["id"]: {value["id"] for value in status["values"]} for status in DEFINITIONS["status_definitions"]}
    assert set(PRESENTATION["statuses"]) == set(defined)
    for status_id, values in PRESENTATION["statuses"].items():
        assert set(values) == defined[status_id], status_id
        assert all(value["label"] for value in values.values())


def test_family_wording_matches_defined_families():
    assert set(PRESENTATION["families"]) == {family["id"] for family in DEFINITIONS["evaluation_families"]}


def test_confusion_categories_match_the_scored_matrix():
    summary = aggregate_scores([score_compliance("PASS", "PASS")])
    assert set(PRESENTATION["confusion_categories"]) == set(summary["overall"]["compliance"]["binary"]["confusion_matrix"])


def test_run_statuses_match_the_runner():
    ok, failed = {"status": "success"}, {"status": "graph_exception"}
    produced = {run_status([ok], []), run_status([ok], [{"category": "x"}]), run_status([failed], [])}
    assert set(PRESENTATION["run_status"]) == produced


def test_error_categories_exist_in_the_engine():
    engine = EVALUATION_ROOT / "fyp_evaluation_engine"
    source = (engine / "adapter.py").read_text(encoding="utf-8") + (engine / "runner.py").read_text(encoding="utf-8")
    for category in PRESENTATION["error_categories"]:
        assert f'"{category}"' in source, category


def test_operation_rows_are_worded_for_every_comparison_row():
    from dashboard.presenters import OPERATION_IDS

    assert set(PRESENTATION["operations"]) == set(OPERATION_IDS)
    for key, entry in PRESENTATION["operations"].items():
        assert entry["label"] and entry["what"] and entry["calculation"], key
        for field in ("label", "what", "calculation", "important"):
            assert not TECHNICAL_TERMS.search(entry.get(field, "")), (key, field)


def test_every_extraction_field_in_the_shipped_datasets_has_a_readable_label():
    fields = set()
    for checklist in (EVALUATION_ROOT / "datasets").glob("*/checklist.json"):
        fields |= {item["field"] for item in json.loads(checklist.read_text(encoding="utf-8-sig")) if item.get("field")}
    assert fields, "expected extraction fields in the pilot dataset"
    assert fields <= set(PRESENTATION["fields"]), sorted(fields - set(PRESENTATION["fields"]))
    for field, entry in PRESENTATION["fields"].items():
        assert entry["label"] and "_" not in entry["label"], field


def test_breach_recall_explains_its_use_not_the_denominator_caveat():
    note = PRESENTATION["metrics"]["compliance.breach_recall"]["important"]
    assert "missed" in note.lower() and "UNCLEAR" not in note
    # The population still appears in the calculation line and in the formal definition.
    assert "PASS/FAIL decision" in PRESENTATION["metrics"]["compliance.breach_recall"]["calculation"]
