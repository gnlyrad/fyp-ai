"""Dashboard API over runs produced by the real runner with an injected graph."""
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.api import create_app
from dashboard.local_repository import LocalEvaluationRepository
from dashboard.repository import RepositoryError
from fyp_evaluation_engine.runner import run_evaluation

EVALUATION_ROOT = Path(__file__).resolve().parents[1]


def _graph(verdicts):
    """Fake graph: per-question verdicts, evidence = the first transcript turn."""
    def invoke(value):
        chunks = [{"chunk_id": f"chunk-{i:02d}", "source_segment_id": s["id"]}
                  for i, s in enumerate(value["results"]["audio_segments"])]
        findings = [{"checklist_id": item["id"], "verdict": verdicts.get(item["id"], "Pass"),
                     "spans": [{"chunk_id": "chunk-00"}], "reason": "Synthetic reasoning."}
                    for item in value["checklist"]]
        return {"chunks": chunks, "report": {"findings": findings, "summary": {}, "warnings": []}}
    return invoke


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluation")
    shutil.copytree(EVALUATION_ROOT / "datasets" / "pilot", root / "datasets" / "pilot")
    for name in ("metric_definitions.json", "metric_presentation.json"):
        shutil.copy2(EVALUATION_ROOT / name, root / name)
    dataset = root / "datasets" / "pilot"
    run_evaluation(dataset, root / "runs", run_id="run-a", graph_invoke=_graph({}))
    run_evaluation(dataset, root / "runs", run_id="run-b", graph_invoke=_graph({"C01": "Fail", "C02": "Unclear"}))
    run_evaluation(dataset, root / "runs", run_id="run-partial", call_ids=["CALL_001"], graph_invoke=_graph({}))
    return root


@pytest.fixture(scope="module")
def client(root):
    def shared():
        raise RepositoryError("Shared source requires SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY.")
    return TestClient(create_app(evaluation_root=root, shared_factory=shared, web_dist=None, default_source="local"))


def _dataset_id(client):
    return client.get("/api/sources/local/datasets").json()[0]["dataset_id"]


def test_meta_serves_definitions_presentation_and_drills(client):
    meta = client.get("/api/meta").json()
    assert meta["default_source"] == "local"
    assert {m["id"] for m in meta["definitions"]["metrics"]} == set(meta["presentation"]["metrics"])
    assert meta["drills"]["confusion.FN"]["filter"] == {"family": "compliance", "confusion_category": "FN"}


def test_datasets_and_run_history(client):
    datasets = client.get("/api/sources/local/datasets").json()
    assert [(d["label"], d["call_count"], d["compliance_questions"], d["extraction_questions"], d["run_count"])
            for d in datasets] == [("Pilot", 5, 6, 4, 3)]
    detail = client.get(f"/api/sources/local/datasets/{datasets[0]['dataset_id']}").json()
    runs = {run["run_id"]: run for run in detail["runs"]}
    assert set(runs) == {"run-a", "run-b", "run-partial"}
    assert runs["run-a"]["is_complete"] and not runs["run-partial"]["is_complete"]
    assert runs["run-a"]["status"]["label"] == "Complete"
    coverage = runs["run-b"]["metrics"]["compliance.decision_coverage"]
    assert (coverage["numerator"], coverage["denominator"]) == (25, 30)


def test_run_returns_typed_rows_with_outcomes_and_problems(client):
    run = client.get("/api/sources/local/runs/run-b").json()
    assert len(run["rows"]) == 50
    assert run["dataset"]["label"] == "Pilot"
    row = next(r for r in run["rows"] if (r["call_id"], r["question_id"]) == ("CALL_001", "C02"))
    assert row["compliance_outcome_status"] == "abstention" and row["outcome"]["label"] == "UNCLEAR"
    assert isinstance(row["gold_turns"], list) and isinstance(row["evidence_precision"], float)
    drills = {problem["drill"]: problem["count"] for problem in run["problems"] if problem["target"] == "cases"}
    assert drills["compliance.abstention"] == 5
    meta = client.get("/api/meta").json()
    for key, count in drills.items():
        row_filter = meta["drills"][key]["filter"]
        assert sum(all(r.get(k) == v for k, v in row_filter.items()) for r in run["rows"]) == count


def test_individual_result_explains_evidence_with_transcript(client):
    result = client.get("/api/sources/local/runs/run-a/results/CALL_001/C01").json()
    assert result["row"]["question_id"] == "C01"
    assert result["question"]["question"]
    assert result["turns"][0]["selected"] is True and result["turns"][0]["speaker"] in {"RM", "Customer"}
    assert result["explanation"].startswith("The model selected")
    assert result["reasoning"] == "Synthetic reasoning."


def test_diagnostics_and_compare(client):
    diagnostics = client.get("/api/sources/local/runs/run-a/diagnostics").json()
    assert diagnostics["errors"] == [] and diagnostics["counts"] == {}
    comparable = client.get("/api/sources/local/compare", params={"a": "run-b", "b": "run-a"}).json()
    assert comparable["comparable"] is True
    # Argument order does not matter: the earlier run is always reported first.
    assert (comparable["earlier"]["run_id"], comparable["later"]["run_id"]) == ("run-a", "run-b")
    assert comparable["unverifiable_run_ids"] == []
    assert {row["id"] for row in comparable["operations"]} == {
        "latency_total", "latency_per_call", "tokens_total", "cost_total", "cost_per_call"}
    latency = next(row for row in comparable["operations"] if row["id"] == "latency_total")
    assert latency["delta"] == pytest.approx(latency["later"] - latency["earlier"])
    tokens = next(row for row in comparable["operations"] if row["id"] == "tokens_total")
    assert tokens["earlier"] is None and tokens["delta"] is None  # injected graph reports no usage
    coverage = next(m for m in comparable["metrics"] if m["metric_id"] == "compliance.decision_coverage")
    assert coverage["delta_pp"] == pytest.approx(-16.67, abs=0.01)
    partial = client.get("/api/sources/local/compare", params={"a": "run-a", "b": "run-partial"}).json()
    assert partial["comparable"] is False


def test_errors_map_to_http_statuses(client):
    assert client.get("/api/sources/local/runs/missing").status_code == 404
    assert client.get("/api/sources/elsewhere/datasets").status_code == 404
    unavailable = client.get("/api/sources/shared/datasets")
    assert unavailable.status_code == 503 and "SUPABASE_URL" in unavailable.json()["detail"]


def test_shared_source_uses_the_same_contract(root):
    client = TestClient(create_app(evaluation_root=root, web_dist=None,
                                   shared_factory=lambda: LocalEvaluationRepository(root)))
    assert client.get("/api/sources/shared/datasets").json()[0]["run_count"] == 3
    assert len(client.get("/api/sources/shared/runs/run-a").json()["rows"]) == 50


def test_built_frontend_is_served_with_spa_fallback(root, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=root></div>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    client = TestClient(create_app(evaluation_root=root, web_dist=dist))
    assert client.get("/local/d/x/r/y?tab=analysis").text == "<div id=root></div>"
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert client.get("/api/nothing").status_code == 404
    assert client.get("/api/meta").status_code == 200
