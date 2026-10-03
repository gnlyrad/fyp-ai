import json
from pathlib import Path

import pytest

from fyp_evaluation_engine.publication import (
    PublicationError,
    SupabasePublisher,
    metric_rows,
    validate_run_dir,
)
from fyp_evaluation_engine.runner import run_evaluation

EVALUATION_ROOT = Path(__file__).resolve().parents[1]


def _invoke(value):
    chunks = [
        {"chunk_id": f"chunk-{index}", "source_segment_id": segment["id"]}
        for index, segment in enumerate(value["results"]["audio_segments"])
    ]
    findings = [
        {
            "checklist_id": item["id"],
            "verdict": "Pass",
            "evidence_status": "absence_after_full_review",
            "spans": [],
            "question_type": "exhaustive",
        }
        for item in value["checklist"]
    ]
    return {"chunks": chunks, "report": {"findings": findings, "summary": {}, "warnings": [], "consistency": {}}}


def test_complete_run_validates_for_publication(tmp_path):
    run_evaluation(EVALUATION_ROOT / "datasets" / "pilot", tmp_path, run_id="complete", graph_invoke=_invoke)
    publication = validate_run_dir(tmp_path / "complete", EVALUATION_ROOT)
    assert publication.metadata["run_id"] == "complete"
    assert len(publication.results) == 50
    assert len(publication.artifact_fingerprint) == 64
    metrics = metric_rows(publication)
    assert any(row["metric_id"] == "compliance.breach_recall" and row["scope_type"] == "overall" for row in metrics)


def test_partial_run_is_rejected(tmp_path):
    run_evaluation(
        EVALUATION_ROOT / "datasets" / "pilot",
        tmp_path,
        run_id="partial",
        call_ids=["CALL_001"],
        graph_invoke=_invoke,
    )
    with pytest.raises(PublicationError, match="Partial-call runs"):
        validate_run_dir(tmp_path / "partial", EVALUATION_ROOT)


def test_complete_run_with_prediction_errors_remains_publishable(tmp_path):
    run_evaluation(EVALUATION_ROOT / "datasets" / "pilot", tmp_path, run_id="with-errors", graph_invoke=_invoke)
    metadata_path = tmp_path / "with-errors" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["execution_status"] = "completed_with_errors"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    publication = validate_run_dir(tmp_path / "with-errors", EVALUATION_ROOT)
    assert publication.metadata["execution_status"] == "completed_with_errors"


def test_changed_run_contents_change_fingerprint(tmp_path):
    run_evaluation(EVALUATION_ROOT / "datasets" / "pilot", tmp_path, run_id="complete", graph_invoke=_invoke)
    before = validate_run_dir(tmp_path / "complete", EVALUATION_ROOT).artifact_fingerprint
    metadata_path = tmp_path / "complete" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["note"] = "changed"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    after = validate_run_dir(tmp_path / "complete", EVALUATION_ROOT).artifact_fingerprint
    assert before != after


def test_identical_published_run_is_an_idempotent_no_op(tmp_path):
    run_evaluation(EVALUATION_ROOT / "datasets" / "pilot", tmp_path, run_id="complete", graph_invoke=_invoke)
    publication = validate_run_dir(tmp_path / "complete", EVALUATION_ROOT)
    publisher = object.__new__(SupabasePublisher)
    publisher._select_one = lambda *_: {
        "run_id": "complete",
        "artifact_fingerprint_sha256": publication.artifact_fingerprint,
        "publication_state": "published",
    }
    assert publisher.publish(publication) == "already_published"


def test_same_run_id_with_different_fingerprint_is_rejected(tmp_path):
    run_evaluation(EVALUATION_ROOT / "datasets" / "pilot", tmp_path, run_id="complete", graph_invoke=_invoke)
    publication = validate_run_dir(tmp_path / "complete", EVALUATION_ROOT)
    publisher = object.__new__(SupabasePublisher)
    publisher._select_one = lambda *_: {
        "run_id": "complete",
        "artifact_fingerprint_sha256": "0" * 64,
        "publication_state": "published",
    }
    with pytest.raises(PublicationError, match="Run-ID integrity collision"):
        publisher.publish(publication)


class _Table:
    def __init__(self, db, name):
        self.db, self.name, self.filters, self.op, self.payload, self.window = db, name, [], "select", None, None

    def select(self, *_):
        self.op = "select"
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def limit(self, _):
        return self

    def range(self, start, end):
        self.window = (start, end)
        return self

    def insert(self, rows):
        self.op, self.payload = "insert", rows if isinstance(rows, list) else [rows]
        return self

    def update(self, values):
        self.op, self.payload = "update", values
        return self

    def execute(self):
        from types import SimpleNamespace
        self.db.requests += 1
        rows = self.db.tables.setdefault(self.name, [])
        if self.op == "insert":
            rows.extend(dict(row) for row in self.payload)
            return SimpleNamespace(data=self.payload)
        matched = [row for row in rows if all(row.get(c) == v for c, v in self.filters)]
        if self.op == "update":
            for row in matched:
                row.update(self.payload)
            return SimpleNamespace(data=matched)
        if self.window:
            matched = matched[self.window[0]:self.window[1] + 1]
        return SimpleNamespace(data=[dict(row) for row in matched])


class _Bucket:
    def __init__(self, db):
        self.db = db

    def upload(self, path, content, file_options=None):
        self.db.requests += 1
        if path in self.db.objects:
            raise RuntimeError("Duplicate")
        self.db.objects[path] = content

    def download(self, path):
        self.db.requests += 1
        return self.db.objects[path]


class _FakeSupabase:
    def __init__(self):
        self.tables, self.objects, self.requests = {}, {}, 0
        self.storage = type("Storage", (), {"from_": lambda _self, _bucket: _Bucket(self)})()

    def table(self, name):
        return _Table(self, name)


def _complete_publication(tmp_path):
    run_evaluation(EVALUATION_ROOT / "datasets" / "pilot", tmp_path, run_id="complete", graph_invoke=_invoke)
    return validate_run_dir(tmp_path / "complete", EVALUATION_ROOT)


def test_publish_batches_requests_and_reports_progress(tmp_path):
    publication = _complete_publication(tmp_path)
    client, steps = _FakeSupabase(), []
    publisher = SupabasePublisher(client, progress=steps.append)
    assert publisher.publish(publication) == "published"
    # Hundreds of rows, but only a handful of requests per table (not two per row).
    assert len(client.tables["evaluation_run_results"]) == 50
    assert len(client.tables["evaluation_run_metrics"]) > 100
    assert client.requests < 120, client.requests
    assert client.tables["evaluation_runs"][0]["publication_state"] == "published"
    assert client.tables["evaluation_datasets"][0]["publication_state"] == "published"
    assert any(step.startswith("Writing 50 results") for step in steps)
    assert steps[-1] == "Marking the run as published"


def test_second_run_on_a_published_dataset_skips_dataset_work(tmp_path):
    first = _complete_publication(tmp_path / "a")
    client = _FakeSupabase()
    SupabasePublisher(client).publish(first)
    uploads_after_first = len(client.objects)
    run_evaluation(EVALUATION_ROOT / "datasets" / "pilot", tmp_path / "b", run_id="second", graph_invoke=_invoke)
    second = validate_run_dir(tmp_path / "b" / "second", EVALUATION_ROOT)
    steps = []
    SupabasePublisher(client, progress=steps.append).publish(second)
    assert "Dataset snapshot already published" in steps
    assert not any(step.startswith("Uploading dataset") for step in steps)
    assert len(client.objects) > uploads_after_first  # only the new run's files were added
    assert len(client.tables["evaluation_runs"]) == 2


def test_interrupted_publication_resumes_and_a_republish_is_a_no_op(tmp_path):
    publication = _complete_publication(tmp_path)
    client = _FakeSupabase()
    publisher = SupabasePublisher(client)
    publisher.publish(publication)
    # Simulate a crash before finalizing: the run is back in 'publishing' with half its rows gone.
    client.tables["evaluation_runs"][0]["publication_state"] = "publishing"
    client.tables["evaluation_run_results"][:] = client.tables["evaluation_run_results"][:20]
    assert publisher.publish(publication) == "published"
    assert len(client.tables["evaluation_run_results"]) == 50
    assert publisher.publish(publication) == "already_published"


def test_conflicting_rows_are_rejected(tmp_path):
    publication = _complete_publication(tmp_path)
    client = _FakeSupabase()
    publisher = SupabasePublisher(client)
    publisher.publish(publication)
    client.tables["evaluation_runs"][0]["publication_state"] = "publishing"
    client.tables["evaluation_run_results"][0]["execution_status"] = "tampered"
    with pytest.raises(PublicationError, match="Database integrity collision in evaluation_run_results"):
        publisher.publish(publication)
