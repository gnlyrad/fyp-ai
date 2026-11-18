"""Speech-to-text API over exported bundles: reads, ids, audio serving (Range, root confinement, hash check) and the wording file."""
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dashboard.api import EVALUATION_ROOT, create_app

AUDIO = b"ID3" + bytes(range(256)) * 40  # stands in for an mp3; the server never decodes it


def unit(uid, op, ref="hello", hyp="hello", **extra):
    base = {"id": uid, "turn_id": 0, "ref_text": ref, "hyp_text": hyp, "op": op, "errors": 0 if op == "equal" else 1,
            "substitutions": int(op == "substitute"), "deletions": int(op == "delete"), "insertions": int(op == "insert"),
            "entity_errors": 0, "entity_groups": [], "parts": [{"ref": ref, "hyp": hyp, "op": op, "entity": False}],
            "ref_role": "EMPLOYEE", "speaker_mismatch": False, "min_confidence": 0.9, "low_confidence": False, "items": [],
            "start": 1.0, "end": 1.5, "time_source": "aws", "approximate": False}
    return {**base, **extra}


def make_bundle(root, run_id="run1", dataset_id="toy_v1", digest=None):
    bundle = root / run_id
    (bundle / "calls").mkdir(parents=True)
    summary = {"calls_scored": 1, "normalizer_version": "1.2.0", "jiwer_version": "4.0.0",
               "normalised_wer": {"reference_words": 2, "substitutions": 1, "deletions": 0, "insertions": 0, "hits": 1, "errors": 1, "wer": 0.5},
               "entity_wer": {"available": False, "reason": "no annotations"},
               "average_confidence": {"items": 2, "mean": 0.9}, "speaker_attribution": {"accuracy": 1.0, "eligible_items": 1, "correct_items": 1}}
    (bundle / "run.json").write_text(json.dumps({
        "schema_version": "1.0", "run_id": run_id, "dataset_id": dataset_id, "render_id": "clean_voice_A", "dataset_call_ids": ["CALL_001", "CALL_002"],
        "collected_at": "2026-10-17T00:00:00Z", "scored_at": "2026-10-17T01:00:00Z", "versions": {"normalizer": "1.2.0", "jiwer": "4.0.0"},
        "provenance": {}, "transcribe_job": {"language_code": "en-US", "vocabulary_name": None}, "summary": summary,
        "calls": [{"call_id": "CALL_001", "wer": summary["normalised_wer"]}]}), encoding="utf-8")
    (bundle / "calls" / "CALL_001.json").write_text(json.dumps({
        "call_id": "CALL_001", "audio": {"dataset_id": dataset_id, "call_id": "CALL_001", "render_id": "clean_voice_A",
                                         "audio_sha256": digest or hashlib.sha256(AUDIO).hexdigest()},
        "metrics": {}, "turns": [], "units": [unit(0, "equal"), unit(1, "substitute", "cat", "dog")], "entity_spans": []}), encoding="utf-8")
    (bundle / "errors.json").write_text("[]", encoding="utf-8")
    (bundle / "unnormalised.json").write_text("{}", encoding="utf-8")
    return bundle


@pytest.fixture(autouse=True)
def no_ambient_audio_root(monkeypatch):
    """Other tests (or a developer's .env) may have set STT_AUDIO_ROOT in this process; each test here sets it explicitly."""
    monkeypatch.delenv("STT_AUDIO_ROOT", raising=False)


@pytest.fixture
def world(tmp_path):
    stt, audio = tmp_path / "stt_runs", tmp_path / "audio"
    make_bundle(stt)
    mp3 = audio / "toy_v1" / "calls" / "CALL_001" / "renders" / "clean_voice_A" / "audio.mp3"
    mp3.parent.mkdir(parents=True)
    mp3.write_bytes(AUDIO)
    client = TestClient(create_app(evaluation_root=EVALUATION_ROOT, web_dist=None, default_source="local", stt_root=stt, stt_audio_root=audio))
    return client, stt, audio, mp3


def test_datasets_runs_run_and_call_reads(world):
    client = world[0]
    datasets = client.get("/api/stt/datasets").json()
    assert datasets == [{"dataset_id": "toy_v1", "label": "Toy V1 (clean)", "run_count": 1, "latest_run_at": "2026-10-17T00:00:00Z", "dataset_call_count": 2}]
    runs = client.get("/api/stt/datasets/toy_v1").json()["runs"]
    assert runs[0]["run_id"] == "run1" and runs[0]["calls_scored"] == 1 and runs[0]["dataset_call_count"] == 2
    run = client.get("/api/stt/runs/run1").json()
    assert run["summary"]["normalised_wer"]["wer"] == 0.5 and run["audio_configured"] is True
    call = client.get("/api/stt/runs/run1/calls/CALL_001").json()
    assert call["audio_status"] == {"available": True, "reason": None} and call["next_call"] is None and len(call["units"]) == 2


@pytest.mark.parametrize("url", ["/api/stt/runs/nope", "/api/stt/runs/run1/calls/CALL_009", "/api/stt/runs/run1/calls/..%2Fx", "/api/stt/datasets/none",
                                 "/api/stt/runs/..%2F..%2Fx"])
def test_unknown_or_malformed_ids_are_404(world, url):
    assert world[0].get(url).status_code == 404


def test_audio_is_served_with_range_support(world):
    client = world[0]
    full = client.get("/api/stt/runs/run1/calls/CALL_001/audio")
    assert full.status_code == 200 and full.content == AUDIO and full.headers["accept-ranges"] == "bytes"
    part = client.get("/api/stt/runs/run1/calls/CALL_001/audio", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == AUDIO[100:200] and part.headers["content-range"] == f"bytes 100-199/{len(AUDIO)}"


def test_audio_is_refused_when_the_recording_is_not_the_evaluated_one(world):
    client, _, _, mp3 = world
    mp3.write_bytes(AUDIO + b"re-rendered")
    response = client.get("/api/stt/runs/run1/calls/CALL_001/audio")
    assert response.status_code == 409 and "not the recording that was evaluated" in response.json()["detail"]
    status = client.get("/api/stt/runs/run1/calls/CALL_001").json()["audio_status"]  # the real reason is known before playback
    assert status["available"] is False and "not the recording that was evaluated" in status["reason"]


def test_audio_is_unavailable_without_a_root_or_a_file_and_never_leaves_the_root(world, tmp_path):
    client, stt, _, mp3 = world
    mp3.unlink()
    missing = client.get("/api/stt/runs/run1/calls/CALL_001/audio")
    assert missing.status_code == 404 and "Audio unavailable on this machine" in missing.json()["detail"]
    assert client.get("/api/stt/runs/run1/calls/CALL_001").json()["audio_status"]["available"] is False
    turned_off = TestClient(create_app(evaluation_root=EVALUATION_ROOT, web_dist=None, stt_root=stt, stt_audio_root=""))
    assert "playback is turned off" in turned_off.get("/api/stt/runs/run1/calls/CALL_001/audio").json()["detail"]
    # a bundle trying to point outside the root through its ids is refused
    outside = tmp_path / "secret.mp3"
    outside.write_bytes(AUDIO)
    call_file = stt / "run1" / "calls" / "CALL_001.json"
    call = json.loads(call_file.read_text(encoding="utf-8"))
    call["audio"]["dataset_id"] = ".."
    call_file.write_text(json.dumps(call), encoding="utf-8")
    assert client.get("/api/stt/runs/run1/calls/CALL_001/audio").status_code == 404


def test_the_run_carries_the_scoring_rule_change_report_when_the_bundle_has_one(world):
    client, stt, _, _ = world
    assert client.get("/api/stt/runs/run1").json()["changes"] is None
    report = {"from": {"normalizer_version": "1.2.0"}, "to": {"normalizer_version": "1.3.0"}, "by_rule": {"dates": {"errors_removed_when_all_rules_on": 2}}}
    (stt / "run1" / "changes_from_1.2.0.json").write_text(json.dumps(report), encoding="utf-8")
    assert client.get("/api/stt/runs/run1").json()["changes"] == report


def test_run_words_lists_flagged_words_and_counts_each_item_once(world):
    client, stt, _, _ = world
    call_file = stt / "run1" / "calls" / "CALL_001.json"
    call = json.loads(call_file.read_text(encoding="utf-8"))
    item = lambda n, conf, eligible: {"item": n, "content": f"w{n}", "confidence": conf, "speaker": "spk_1", "start": n, "end": n + 0.4,  # noqa: E731
                                      "mapped_role": "CLIENT", "eligible": eligible}
    shared = item(1, 0.95, True)
    call["units"] = [unit(0, "equal", items=[shared]), unit(1, "equal", "b", "b", items=[shared]),  # one item behind two Gold words
                     unit(2, "substitute", "cat", "dog", items=[item(2, 0.3, False)], low_confidence=True, min_confidence=0.3),
                     unit(3, "equal", "yes", "yes", items=[item(3, 0.6, True)], speaker_mismatch=True, min_confidence=0.6)]
    call_file.write_text(json.dumps(call), encoding="utf-8")
    words = client.get("/api/stt/runs/run1/words").json()
    assert [(b["from"], b["matched"], b["other"]) for b in words["confidence_buckets"]] == [(0.0, 0, 1), (0.5, 1, 0), (0.7, 0, 0), (0.9, 1, 0)]
    assert [(w["unit"], w["min_confidence"], w["errors"]) for w in words["low_confidence"]] == [(2, 0.3, 1), (3, 0.6, 0)]  # everything below 0.9, lowest first
    assert words["entity_spans"] == []
    assert [(w["call_id"], w["unit"], w["ref_role"]) for w in words["speaker_mismatch"]] == [("CALL_001", 3, "EMPLOYEE")]
    assert client.get("/api/stt/runs/nope/words").status_code == 404


def test_run_words_lists_entity_spans_with_the_unit_to_open(world):
    client, stt, _, _ = world
    call_file = stt / "run1" / "calls" / "CALL_001.json"
    call = json.loads(call_file.read_text(encoding="utf-8"))
    call["units"] = [unit(0, "equal"), unit(1, "substitute", "cat", "dog", entity_groups=["value:T03"], turn_id=2)]
    call["entity_spans"] = [{"turn_id": 2, "group": "value:T03", "text": "cat", "errors": 1, "tokens": ["cat"], "hypothesis_tokens": ["dog"]},
                            {"turn_id": 5, "group": "value:K01", "text": "x", "errors": 0, "tokens": ["x"], "hypothesis_tokens": ["x"]}]
    call_file.write_text(json.dumps(call), encoding="utf-8")
    spans = client.get("/api/stt/runs/run1/words").json()["entity_spans"]
    assert [(s["group"], s["unit"], s["errors"], s["hypothesis_tokens"]) for s in spans] == [("value:T03", 1, 1, ["dog"]), ("value:K01", None, 0, ["x"])]


def test_the_wording_file_covers_every_metric_the_run_page_shows():
    wording = json.loads((EVALUATION_ROOT / "stt_presentation.json").read_text(encoding="utf-8"))
    for metric in ("normalised_wer", "entity_wer", "substitutions", "deletions", "insertions", "average_confidence", "speaker_attribution"):
        entry = wording["metrics"][metric]
        assert entry["label"] and entry["what"] and entry["calculation"] and "important" in entry
    assert {"legend", "highlight_note", "approximate_time"} <= set(wording["diff"]) and wording["audio"]["unavailable"]
    client = TestClient(create_app(evaluation_root=EVALUATION_ROOT, web_dist=None))
    assert client.get("/api/meta").json()["stt_presentation"]["tab_label"] == "Speech-to-text"


REAL = sorted((EVALUATION_ROOT / "stt_runs").glob("*/run.json"))


@pytest.mark.skipif(not REAL, reason="no exported speech-to-text runs on this machine")
@pytest.mark.parametrize("run_json", REAL)
def test_real_bundles_display_exactly_the_exported_scores(run_json):
    client = TestClient(create_app(evaluation_root=EVALUATION_ROOT, web_dist=None))
    run = client.get(f"/api/stt/runs/{run_json.parent.name}").json()
    total = {"substitutions": 0, "deletions": 0, "insertions": 0}
    for row in run["calls"]:
        call = client.get(f"/api/stt/runs/{run_json.parent.name}/calls/{row['call_id']}").json()
        shown = {key: sum(u[key] for u in call["units"]) for key in total}
        assert shown == {key: row["wer"][key] for key in total}, row["call_id"]
        for key in total:
            total[key] += shown[key]
    assert total == {key: run["summary"]["normalised_wer"][key] for key in total}


def test_a_relative_audio_root_is_resolved_against_the_repository_root_not_the_working_directory(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    (repository / "evaluation").mkdir(parents=True)
    stt = repository / "evaluation" / "stt_runs"
    make_bundle(stt)
    mp3 = tmp_path / "be" / "datasets" / "toy_v1" / "calls" / "CALL_001" / "renders" / "clean_voice_A" / "audio.mp3"
    mp3.parent.mkdir(parents=True)
    mp3.write_bytes(AUDIO)
    monkeypatch.chdir(tmp_path / "be")  # a different working directory must not matter
    client = TestClient(create_app(evaluation_root=repository / "evaluation", web_dist=None, stt_audio_root="../be/datasets"))
    assert client.get("/api/stt/runs/run1/calls/CALL_001").json()["audio_status"] == {"available": True, "reason": None}
    assert client.get("/api/stt/runs/run1/calls/CALL_001/audio").content == AUDIO


def sibling_checkouts(parent):
    """<parent>/fyp-ai/evaluation with one bundle, and the recording in <parent>/fyp-be/evaluation-dataset/datasets."""
    evaluation = parent / "fyp-ai" / "evaluation"
    evaluation.mkdir(parents=True)
    make_bundle(evaluation / "stt_runs")
    mp3 = parent / "fyp-be" / "evaluation-dataset" / "datasets" / "toy_v1" / "calls" / "CALL_001" / "renders" / "clean_voice_A" / "audio.mp3"
    mp3.parent.mkdir(parents=True)
    mp3.write_bytes(AUDIO)
    return evaluation, mp3


def test_without_any_setting_audio_comes_from_the_fyp_be_checkout_beside_fyp_ai(tmp_path, monkeypatch):
    evaluation, _ = sibling_checkouts(tmp_path / "anywhere")
    monkeypatch.chdir(tmp_path)  # the working directory must not matter
    client = TestClient(create_app(evaluation_root=evaluation, web_dist=None))
    assert client.get("/api/stt/runs/run1/calls/CALL_001").json()["audio_status"] == {"available": True, "reason": None}
    assert client.get("/api/stt/runs/run1/calls/CALL_001/audio").content == AUDIO


def test_stt_audio_root_overrides_the_default_and_is_named_in_reasons(tmp_path, monkeypatch):
    evaluation, _ = sibling_checkouts(tmp_path)
    monkeypatch.setenv("STT_AUDIO_ROOT", "../elsewhere")
    status = TestClient(create_app(evaluation_root=evaluation, web_dist=None)).get("/api/stt/runs/run1/calls/CALL_001").json()["audio_status"]
    assert status["available"] is False and "STT_AUDIO_ROOT=../elsewhere does not exist" in status["reason"]
    monkeypatch.delenv("STT_AUDIO_ROOT")
    (tmp_path / "fyp-be").rename(tmp_path / "moved")
    status = TestClient(create_app(evaluation_root=evaluation, web_dist=None)).get("/api/stt/runs/run1/calls/CALL_001").json()["audio_status"]
    assert "../fyp-be/evaluation-dataset/datasets (default: fyp-be beside fyp-ai) does not exist" in status["reason"]


def test_each_unavailable_reason_is_reported_on_the_call(world, tmp_path):
    _, stt, _, mp3 = world
    def reason(root):
        client = TestClient(create_app(evaluation_root=EVALUATION_ROOT, web_dist=None, stt_root=stt, stt_audio_root=root))
        status = client.get("/api/stt/runs/run1/calls/CALL_001").json()["audio_status"]
        assert status["available"] is False
        return status["reason"]
    assert "playback is turned off" in reason("")
    assert f"the audio folder {tmp_path / 'nowhere'} does not exist" in reason(str(tmp_path / "nowhere"))
    mp3.unlink()
    assert "no recording at toy_v1/calls/CALL_001/renders/clean_voice_A/audio.mp3" in reason(str(tmp_path / "audio"))
