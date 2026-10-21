import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fyp_evaluation_loader import DatasetError, graph_input_for_call, load_dataset
from fyp_evaluation_engine.current_pipeline import prepare_input


DATASET = ROOT / "datasets" / "pilot"


def test_every_published_call_prepares_direct_graph_input():
    dataset = load_dataset(DATASET)
    assert dataset["manifest"]["call_ids"] == ["CALL_001", "CALL_002", "CALL_003", "CALL_004", "CALL_005"]
    for call_id in dataset["manifest"]["call_ids"]:
        value = graph_input_for_call(dataset, call_id)
        assert set(value) == {"results", "speaker_roles", "checklist"}
        assert value["results"]["audio_segments"]
        source = {item["id"]: item for item in dataset["checklist"]}
        for item in value["checklist"]:
            if source[item["id"]]["type"] == "COMPLIANCE":
                assert set(item) == {"id", "question", "requirement_type"}
                assert item["requirement_type"] == source[item["id"]]["requirement_type"]
            else:
                assert set(item) == {"id", "question"}
        assert prepare_input(value) == value


def test_evaluation_family_comes_from_checklist_type_and_never_reaches_the_graph():
    from fyp_evaluation_loader import evaluation_family_for_question_type

    dataset = load_dataset(DATASET)
    families = {item["id"]: item["evaluation_family"] for item in dataset["checklist"]}
    assert families == {**{f"C0{n}": "compliance" for n in range(1, 7)},
                        "K01": "extraction", "K02": "extraction", "T01": "extraction", "T02": "extraction"}
    assert [evaluation_family_for_question_type(t) for t in ("COMPLIANCE", "CLIENT_KYC", "TRANSACTION")] == [
        "compliance", "extraction", "extraction"]
    with pytest.raises(DatasetError):
        evaluation_family_for_question_type("OTHER")
    assert all("evaluation_family" not in item for item in dataset["graph_checklist"])


def test_tampered_dataset_is_rejected(tmp_path):
    import shutil
    copied = tmp_path / "pilot"
    shutil.copytree(DATASET, copied)
    path = copied / "calls/CALL_001/transcript.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["audio_segments"][0]["transcript"] = "Modified."
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(DatasetError, match="differs from manifest"):
        load_dataset(copied)
