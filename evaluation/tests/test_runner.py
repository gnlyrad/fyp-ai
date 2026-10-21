import json
import pytest
from pathlib import Path

from fyp_evaluation_engine.runner import run_evaluation

EVALUATION_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_options_reach_graph_and_are_recorded(tmp_path):
    inputs = []

    def invoke(value):
        inputs.append(value)
        return {"chunks": [], "report": {"findings": [
            {"checklist_id": item["id"], "verdict": None, "spans": [],
             "execution_status": "prediction_failure", "failure_category": "analysis_unavailable",
             "evidence_status": "analysis_unavailable", "failure_stage": "judge"}
            for item in value["checklist"]]}}

    result = run_evaluation(EVALUATION_ROOT / "datasets" / "pilot", tmp_path,
                            call_ids=["CALL_001"], graph_invoke=invoke,
                            context_mode="full_transcript", analysis_max_attempts=1)
    assert inputs[0]["context_mode"] == "full_transcript"
    assert inputs[0]["analysis_max_attempts"] == 1
    assert all(set(item) == ({"id", "question", "requirement_type"}
                            if item["id"].startswith("C") else {"id", "question"})
               for item in inputs[0]["checklist"])
    config = result["metadata"]["execution_configuration"]
    assert config["context_mode"] == "full_transcript" and config["analysis_max_attempts"] == 1
    assert result["metadata"]["execution_status"] == "completed_with_errors"
    compliance = result["summary"]["overall"]["compliance"]
    assert compliance["outcome_population"]["prediction_failures"] == 6
    assert compliance["outcome_population"]["abstentions"] == 0


@pytest.mark.parametrize("options", [{"context_mode": "typo"}, {"analysis_max_attempts": 0},
                                     {"analysis_max_attempts": True}])
def test_invalid_experiment_options_do_not_invoke_graph(tmp_path, options):
    def invoke(value):
        pytest.fail("Invalid options must not invoke the graph")
    with pytest.raises(ValueError):
        run_evaluation(EVALUATION_ROOT / "datasets" / "pilot", tmp_path, graph_invoke=invoke, **options)


def test_runner_invokes_graph_once_for_complete_call_and_persists(tmp_path):
    inputs = []

    def invoke(value):
        inputs.append(value)
        chunks = [
            {"chunk_id": f"chunk-{index:02d}", "source_segment_id": segment["id"]}
            for index, segment in enumerate(value["results"]["audio_segments"])
        ]
        findings = [
            {"checklist_id": item["id"], "verdict": "Pass", "evidence_status": "absence_after_full_review",
             "spans": [], "question_type": "exhaustive"}
            for item in value["checklist"]
        ]
        return {"chunks": chunks, "report": {"findings": findings, "summary": {}, "warnings": [], "consistency": {}}}

    result = run_evaluation(
        EVALUATION_ROOT / "datasets" / "pilot", tmp_path,
        call_ids=["CALL_001"], run_id="test-run", graph_invoke=invoke,
    )
    assert len(inputs) == 1
    assert len(inputs[0]["checklist"]) == 10
    assert len(inputs[0]["results"]["audio_segments"]) > 1
    assert result["metadata"]["execution_configuration"]["graph_invocations_per_call"] == 1
    assert result["metadata"]["execution_configuration"]["applicability_bypassed"] is True
    lines = (tmp_path / "test-run" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    assert {json.loads(line)["question_id"] for line in lines} == {item["id"] for item in inputs[0]["checklist"]}


def test_runner_persists_family_aware_records_and_v2_versions(tmp_path):
    def invoke(value):
        chunks = [{"chunk_id": f"chunk-{i:02d}", "source_segment_id": s["id"]}
                  for i, s in enumerate(value["results"]["audio_segments"])]
        findings = [
            {"checklist_id": item["id"], "verdict": "Unclear", "evidence_status": "absence_after_full_review",
             "spans": [], "question_type": "exhaustive"}
            for item in value["checklist"]
        ]
        return {"chunks": chunks, "report": {"findings": findings, "summary": {}, "warnings": [], "consistency": {}}}

    result = run_evaluation(EVALUATION_ROOT / "datasets" / "pilot", tmp_path, call_ids=["CALL_001"],
                            run_id="v2-run", graph_invoke=invoke)
    metadata = json.loads((tmp_path / "v2-run" / "metadata.json").read_text(encoding="utf-8"))
    assert (metadata["artifact_schema_version"], metadata["run_kind"]) == ("2.0.0", "graph_execution")
    assert metadata["dataset_snapshot"]["checklist_sha256"]
    assert len(metadata["metric_definitions_sha256"]) == 64
    assert "git_commit_author" in metadata and "git_commit_timestamp" in metadata
    scores = [json.loads(line) for line in (tmp_path / "v2-run" / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [score["evaluation_family"] for score in scores] == ["compliance"] * 6 + ["extraction"] * 4
    summary = result["summary"]["overall"]
    assert summary["compliance"]["outcome_population"]["abstentions"] == 6
    assert summary["extraction"]["counts"]["not_found"] == 4
    assert summary["counts"]["native_unclear_reconciliation"]["total_native_unclear"] == 10


def test_evaluation_records_gemini_not_bedrock(monkeypatch):
    from fyp_evaluation_engine.runner import _model_configuration
    for name in ("CLEARCALL_MODEL_ID", "GOOGLE_EMBEDDING_MODEL", "CLEARCALL_TEMPERATURE"):
        monkeypatch.delenv(name, raising=False)
    config = _model_configuration()
    assert config["model_provider"] == "google_genai"
    assert config["chat_model_id"] == "gemini-3.5-flash"
    assert config["embedding_model_id"] == "gemini-embedding-001"
    assert config["temperature"] == 1
    assert "aws_region" not in config


def test_pipeline_diagnostics_are_attributed_to_checklist_questions():
    from fyp_evaluation_engine.runner import normalize_and_score_call
    from fyp_evaluation_loader import call_documents, load_dataset

    dataset = load_dataset(EVALUATION_ROOT / "datasets" / "pilot")
    native = {"final_results": [], "segments": [], "errors": [
        "check_item[C01]: resolve_evidence: quote not found in 'seg_0003'; used whole segment, unverified",
        "verify[K02]: model returned no verdict",
        "check_item[Z99]: unknown item id",
        "call_intent: model unavailable",
    ]}
    _, errors, _ = normalize_and_score_call(
        "CALL_001", dataset["checklist"], call_documents(dataset, "CALL_001"), native, 1.0, None
    )
    diagnostics = [error for error in errors if error["category"] == "pipeline_diagnostic"]
    assert [error["question_id"] for error in diagnostics] == ["C01", "K02", None, None]
    assert [error["message"] for error in diagnostics] == native["errors"]
