"""The evidence drill-down must explain the scored evidence numbers, not invent its own."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.api import create_app
from dashboard.local_repository import LocalEvaluationRepository

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
SAVED_RUN = EVALUATION_ROOT / "runs" / "20261003T154136Z-e9fdc884"


def _check_entries(entries, transcript_by_call):
    assert entries, "expected some imperfect evidence"
    for entry in entries:
        gold, predicted = set(entry["gold_turns"]), set(entry["predicted_turns"])
        # The parts are exactly the set differences of the scored turn sets.
        assert [turn["id"] for turn in entry["missed"]] == sorted(gold - predicted)
        assert [turn["id"] for turn in entry["extra"]] == sorted(predicted - gold)
        # ... with the real transcript text attached.
        text = transcript_by_call[entry["call_id"]]
        for turn in (*entry["missed"], *entry["extra"]):
            assert turn["text"] == text[turn["id"]] and turn["speaker"]
        # ... and they explain the scores (standard formulas apply when both sets are non-empty).
        if gold and predicted and entry["execution_status"] == "success":
            assert entry["precision"] == pytest.approx((len(predicted) - len(entry["extra"])) / len(predicted))
            assert entry["recall"] == pytest.approx((len(gold) - len(entry["missed"])) / len(gold))
            assert (entry["exact"] == 1.0) == (not entry["missed"] and not entry["extra"])


@pytest.fixture(scope="module")
def client_and_root(tmp_path_factory):
    from tests.test_dashboard_api import _graph  # reuse the fake graph that cites only the first turn
    import shutil
    from fyp_evaluation_engine.runner import run_evaluation

    root = tmp_path_factory.mktemp("evaluation")
    shutil.copytree(EVALUATION_ROOT / "datasets" / "pilot", root / "datasets" / "pilot")
    for name in ("metric_definitions.json", "metric_presentation.json"):
        shutil.copy2(EVALUATION_ROOT / name, root / name)
    run_evaluation(root / "datasets" / "pilot", root / "runs", run_id="run-e", graph_invoke=_graph({}))
    return TestClient(create_app(evaluation_root=root, web_dist=None, default_source="local")), root


def test_parts_are_the_set_differences_with_text_and_explain_the_scores(client_and_root):
    client, root = client_and_root
    entries = client.get("/api/sources/local/runs/run-e/evidence").json()["entries"]
    repository = LocalEvaluationRepository(root)
    transcripts = {call: {s["id"]: s["transcript"] for s in repository.get_call_transcript("run-e", call)[0]["audio_segments"]}
                   for call in {entry["call_id"] for entry in entries}}
    _check_entries(entries, transcripts)
    assert any(entry["extra"] for entry in entries) and any(entry["missed"] for entry in entries)


def test_perfect_checks_are_left_out(client_and_root):
    client, _ = client_and_root
    entries = client.get("/api/sources/local/runs/run-e/evidence").json()["entries"]
    perfect = [e for e in entries if e["exact"] == 1.0 and e["precision"] == 1.0 and e["recall"] == 1.0
               and not e["invalid_reference_count"] and e["native_verdict"] != "Unclear"]
    assert perfect == []


def test_unknown_run_is_a_404(client_and_root):
    client, _ = client_and_root
    assert client.get("/api/sources/local/runs/missing/evidence").status_code == 404


@pytest.mark.skipif(not SAVED_RUN.is_dir(), reason="saved run not present (runs/ is not committed)")
def test_saved_real_run_drilldown_matches_its_scores():
    client = TestClient(create_app(web_dist=None, default_source="local"))
    entries = client.get(f"/api/sources/local/runs/{SAVED_RUN.name}/evidence").json()["entries"]
    scores = [json.loads(line) for line in (SAVED_RUN / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    with_extra = {(s["call_id"], s["question_id"]) for s in scores
                  if set(s["predicted_evidence_turn_ids"]) - set(s["gold_evidence_turn_ids"])}
    with_missed = {(s["call_id"], s["question_id"]) for s in scores
                   if set(s["gold_evidence_turn_ids"]) - set(s["predicted_evidence_turn_ids"])}
    assert {(e["call_id"], e["question_id"]) for e in entries if e["extra"]} == with_extra
    assert {(e["call_id"], e["question_id"]) for e in entries if e["missed"]} == with_missed
