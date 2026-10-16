import csv
import json
import shutil
from pathlib import Path

from dashboard import LocalEvaluationRepository

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
PILOT = EVALUATION_ROOT / "datasets" / "pilot"


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_evaluation_root(tmp_path):
    root = tmp_path / "evaluation"
    shutil.copytree(PILOT, root / "datasets" / "pilot")
    (root / "runs" / "test-run" / "raw").mkdir(parents=True)
    shutil.copy2(EVALUATION_ROOT / "metric_definitions.json", root / "metric_definitions.json")
    dataset_hash = json.loads((PILOT / "manifest.json").read_text(encoding="utf-8"))["dataset_sha256"]
    metadata = {
        "run_id": "test-run",
        "execution_status": "completed",
        "started_at": "2026-10-03T00:00:00Z",
        "dataset_snapshot": {"dataset_sha256": dataset_hash, "call_ids": ["CALL_001"]},
        "execution_configuration": {"selected_call_ids": ["CALL_001"]},
    }
    _write_json(root / "runs" / "test-run" / "metadata.json", metadata)
    _write_json(root / "runs" / "test-run" / "summary.json", {"overall": {}, "by_call_id": {}, "by_question_id": {}})
    with (root / "runs" / "test-run" / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["call_id", "question_id", "evaluation_family"])
        writer.writeheader()
        writer.writerow({"call_id": "CALL_001", "question_id": "C01", "evaluation_family": "compliance"})
    prediction = {"call_id": "CALL_001", "question_id": "C01", "evaluation_family": "compliance"}
    score = {**prediction, "evidence": {}}
    (root / "runs" / "test-run" / "predictions.jsonl").write_text(json.dumps(prediction) + "\n", encoding="utf-8")
    (root / "runs" / "test-run" / "scores.jsonl").write_text(json.dumps(score) + "\n", encoding="utf-8")
    (root / "runs" / "test-run" / "errors.jsonl").write_text("", encoding="utf-8")
    return root


def test_local_repository_discovers_dataset_partial_run_and_result(tmp_path):
    root = _make_evaluation_root(tmp_path)
    repository = LocalEvaluationRepository(root)
    datasets = repository.list_datasets()
    assert len(datasets) == 1 and datasets[0].name == "pilot"
    runs = repository.list_runs(datasets[0].dataset_id)
    assert len(runs) == 1 and runs[0].is_complete is False
    assert repository.list_results("test-run", family="compliance")[0]["question_id"] == "C01"
    detail = repository.get_result("test-run", "CALL_001", "C01")
    assert detail.question["question"].startswith("The employee")
    assert detail.ground_truth["expected"]["expected_verdict"] == "PASS"
    assert detail.transcript["audio_segments"]


def test_local_repository_metric_definitions_are_available(tmp_path):
    repository = LocalEvaluationRepository(_make_evaluation_root(tmp_path))
    definitions = repository.get_metric_definitions()
    assert definitions["schema_version"] == "2.0.0"


def test_a_run_on_a_missing_dataset_version_is_reported_not_silently_dropped(tmp_path):
    root = _make_evaluation_root(tmp_path)
    repository = LocalEvaluationRepository(root)
    dataset_id = repository.list_datasets()[0].dataset_id
    assert [run.run_id for run in repository.list_runs(dataset_id)] == ["test-run"]
    assert repository.list_unlinked_runs() == []

    # A run made on an older dataset version whose files are no longer in datasets/.
    old = root / "runs" / "old-run"
    shutil.copytree(root / "runs" / "test-run", old)
    metadata = json.loads((old / "metadata.json").read_text(encoding="utf-8"))
    metadata["run_id"] = "old-run"
    metadata["dataset_snapshot"]["dataset_sha256"] = "d138f0fd" + "0" * 56
    (old / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    repository = LocalEvaluationRepository(root)
    assert [run.run_id for run in repository.list_runs(dataset_id)] == ["test-run"]
    assert repository.list_unlinked_runs() == [{
        "run_id": "old-run", "evaluated_at": "2026-10-03T00:00:00Z",
        "dataset_sha256": "d138f0fd" + "0" * 56, "reason": "dataset_missing",
    }]


def test_a_run_still_being_written_is_not_reported_as_broken(tmp_path):
    root = _make_evaluation_root(tmp_path)
    (root / "runs" / "in-progress").mkdir()
    (root / "runs" / "in-progress" / "metadata.json").write_text("{}", encoding="utf-8")  # no summary yet
    assert LocalEvaluationRepository(root).list_unlinked_runs() == []
