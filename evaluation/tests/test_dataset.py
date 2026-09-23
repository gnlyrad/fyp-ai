import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fyp_evaluation_loader import DatasetError, graph_input_for_call, load_dataset


DATASET = ROOT / "datasets" / "pilot"


def test_every_published_call_prepares_direct_graph_input():
    dataset = load_dataset(DATASET)
    assert dataset["manifest"]["call_ids"] == ["CALL_001", "CALL_002", "CALL_003", "CALL_004", "CALL_005"]
    for call_id in dataset["manifest"]["call_ids"]:
        value = graph_input_for_call(dataset, call_id)
        assert set(value) == {"results", "speaker_roles", "checklist"}
        assert value["results"]["audio_segments"]
        assert all(set(item) == {"id", "question"} for item in value["checklist"])


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
