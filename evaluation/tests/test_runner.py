import json
from pathlib import Path

from fyp_evaluation_engine.runner import run_evaluation

EVALUATION_ROOT = Path(__file__).resolve().parents[1]


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
    scores = [json.loads(line) for line in (tmp_path / "v2-run" / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [score["evaluation_family"] for score in scores] == ["compliance"] * 6 + ["extraction"] * 4
    summary = result["summary"]["overall"]
    assert summary["compliance"]["outcome_population"]["abstentions"] == 6
    assert summary["extraction"]["counts"]["not_found"] == 4
    assert summary["counts"]["native_unclear_reconciliation"]["total_native_unclear"] == 10
