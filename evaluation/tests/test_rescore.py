import json
from pathlib import Path

import pytest

from fyp_evaluation_engine.rescore import rescore_run
from fyp_evaluation_engine.runner import run_evaluation
from fyp_evaluation_loader import load_dataset

DATASET = Path(__file__).resolve().parents[1] / "datasets" / "pilot"


def _invoke(value):
    segments = value["results"]["audio_segments"]
    chunks = [{"chunk_id": f"chunk-{i:02d}", "source_segment_id": s["id"]} for i, s in enumerate(segments)]
    findings = []
    for item in value["checklist"]:
        finding = {"checklist_id": item["id"], "verdict": "Pass", "evidence_status": "direct_evidence",
                   "spans": [], "question_type": "deterministic"}
        if item["id"] == "C01":  # caught failure with an UNCLEAR fallback
            finding.update(verdict="Unclear", evidence_status="analysis_unavailable")
        if item["id"] == "C02":  # genuine abstention
            finding.update(verdict="Unclear", evidence_status="absence_after_full_review")
        if item["id"] == "K01":  # successful extraction, native Unclear
            finding.update(verdict="Unclear", spans=[{"chunk_id": "chunk-00", "text": "no identifier here"}])
        findings.append(finding)
    return {"chunks": chunks, "report": {"findings": findings, "summary": {}, "warnings": [], "consistency": {}}}


@pytest.fixture
def source(tmp_path):
    run_evaluation(DATASET, tmp_path, call_ids=["CALL_001", "CALL_002"], run_id="source", graph_invoke=_invoke)
    return tmp_path / "source"


def _snapshot(directory: Path):
    return {path.relative_to(directory).as_posix(): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


def test_rescore_creates_a_new_run_with_lineage_and_never_touches_the_source(source, tmp_path):
    before = _snapshot(source)
    result = rescore_run(source, tmp_path, DATASET, run_id="derived")
    assert _snapshot(source) == before
    derived = tmp_path / "derived"
    assert result["run_dir"] == derived.resolve()
    metadata = json.loads((derived / "metadata.json").read_text(encoding="utf-8"))
    lineage = metadata["derived_from"]
    source_metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_kind"] == "rescored" and metadata["run_id"] == "derived"
    assert lineage["run_id"] == "source" and lineage["graph_invoked"] is False
    assert lineage["dataset_sha256"] == source_metadata["dataset_snapshot"]["dataset_sha256"]
    assert lineage["graph_commit"] == source_metadata["graph_commit"] == metadata["graph_commit"]
    assert (lineage["scorer_version"], metadata["scorer_version"]) == (
        source_metadata["scorer_version"], "fyp-nlp-evaluator-v2.0")
    assert (derived / "raw" / "CALL_001.json").read_bytes() == (source / "raw" / "CALL_001.json").read_bytes()


def test_rescore_reproduces_live_scoring_and_separates_unclear_meanings(source, tmp_path):
    rescore_run(source, tmp_path, DATASET, run_id="derived")
    live = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    derived = json.loads((tmp_path / "derived" / "summary.json").read_text(encoding="utf-8"))
    assert derived == live
    for name in ("scores.jsonl", "predictions.jsonl", "errors.jsonl", "results.csv"):
        assert (tmp_path / "derived" / name).read_bytes() == (source / name).read_bytes()
    reconciliation = derived["overall"]["counts"]["native_unclear_reconciliation"]
    assert reconciliation == {
        "total_native_unclear": 6, "compliance_abstentions": 2, "compliance_prediction_failures": 2,
        "extraction_successful_outputs": 2, "extraction_prediction_failures": 0,
    }


def test_rescore_upgrades_a_legacy_source_run(source, tmp_path):
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("run_kind")
    metadata.update(scorer_version="fyp-nlp-evaluator-v1.1", artifact_schema_version="1.0.0")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result = rescore_run(source, tmp_path, DATASET, run_id="derived")
    lineage = result["metadata"]["derived_from"]
    assert (lineage["run_kind"], lineage["artifact_schema_version"]) == ("graph_execution", "1.0.0")
    assert result["metadata"]["artifact_schema_version"] == "2.0.0"


def test_rescore_handles_a_failed_call(tmp_path):
    def broken(value):
        raise RuntimeError("bedrock unavailable")

    run_evaluation(DATASET, tmp_path, call_ids=["CALL_001"], run_id="source", graph_invoke=broken)
    result = rescore_run(tmp_path / "source", tmp_path, DATASET, run_id="derived")
    counts = result["summary"]["overall"]["counts"]
    assert counts["execution_status"] == {"prediction_failure": 10}
    assert counts["failure_category"] == {"missing_prediction": 10}


def test_rescore_refuses_a_different_dataset_and_an_existing_run(source, tmp_path):
    with pytest.raises(FileExistsError):
        rescore_run(source, tmp_path, DATASET, run_id="source")
    metadata_path = source / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dataset_snapshot"]["dataset_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset_sha256 differs"):
        rescore_run(source, tmp_path, DATASET, run_id="other")
    assert load_dataset(DATASET)  # the dataset itself is untouched
