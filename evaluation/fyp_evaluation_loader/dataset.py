"""Load a published dataset without importing fyp-be or LangGraph."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

FORMAT = "fyp-nlp-dataset-v1"
CALL_FILE_NAMES = ("ground_truth.json", "transcript.json", "speaker_roles.json")
CALL_ID = re.compile(r"CALL_[0-9]{3,}$")


class DatasetError(ValueError):
    """The published dataset is missing, malformed, or internally inconsistent."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise DatasetError(f"Cannot read JSON: {path}") from error


def _canonical_json(data: Any) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _hash(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data)).hexdigest()


def _project_checklist(checklist: Any) -> list[dict[str, str]]:
    if not isinstance(checklist, list) or not checklist:
        raise DatasetError("checklist.json must be a non-empty array")
    projected = []
    seen = set()
    for index, item in enumerate(checklist):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("text"), str):
            raise DatasetError(f"checklist[{index}] needs string id and text")
        if not item["id"] or not item["text"].strip() or item["id"] in seen:
            raise DatasetError(f"checklist[{index}] has an empty or duplicate id/text")
        seen.add(item["id"])
        projected.append({"id": item["id"], "question": item["text"]})
    return projected


def load_dataset(dataset_dir: str | Path) -> dict[str, Any]:
    """Return validated source documents and the graph checklist projection."""
    root = Path(dataset_dir).resolve()
    manifest = _read_json(root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise DatasetError("Unsupported or missing dataset manifest")
    expected_hashes = manifest.get("files")
    call_ids = manifest.get("call_ids")
    if not isinstance(expected_hashes, dict) or not isinstance(call_ids, list) or not call_ids:
        raise DatasetError("Manifest requires files and call_ids")
    if any(not isinstance(call_id, str) or not CALL_ID.fullmatch(call_id) for call_id in call_ids):
        raise DatasetError("Manifest contains an invalid call ID")
    if len(call_ids) != len(set(call_ids)):
        raise DatasetError("Manifest contains duplicate call IDs")
    expected_paths = {"checklist.json"}
    for call_id in call_ids:
        expected_paths.update(f"calls/{call_id}/{name}" for name in CALL_FILE_NAMES)
    if set(expected_hashes) != expected_paths:
        raise DatasetError("Manifest file inventory does not match its call IDs")
    for relative, expected_hash in expected_hashes.items():
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise DatasetError(f"Invalid hash for {relative}")
        actual = _hash(_read_json(root / relative))
        if actual != expected_hash:
            raise DatasetError(f"Dataset file differs from manifest: {relative}")
    checklist = _read_json(root / "checklist.json")
    return {
        "root": root,
        "manifest": manifest,
        "checklist": checklist,
        "graph_checklist": _project_checklist(checklist),
    }


def graph_input_for_call(dataset: dict[str, Any], call_id: str) -> dict[str, Any]:
    """Prepare exactly the direct input consumed by `compliance.graph.invoke`."""
    if call_id not in dataset["manifest"]["call_ids"]:
        raise DatasetError(f"Call is not in this dataset: {call_id}")
    call_dir = dataset["root"] / "calls" / call_id
    ground_truth = _read_json(call_dir / "ground_truth.json")
    transcript = _read_json(call_dir / "transcript.json")
    roles = _read_json(call_dir / "speaker_roles.json")
    if ground_truth.get("call_id") != call_id:
        raise DatasetError(f"{call_id}: ground_truth call_id does not match directory")
    if not isinstance(transcript, dict) or not isinstance(transcript.get("audio_segments"), list):
        raise DatasetError(f"{call_id}: transcript requires audio_segments")
    if not isinstance(roles, dict) or not roles:
        raise DatasetError(f"{call_id}: speaker_roles.json requires a non-empty role mapping")
    labels = {segment.get("speaker_label") for segment in transcript["audio_segments"] if isinstance(segment, dict)}
    if set(roles) != labels or any(role not in {"RM", "Customer"} for role in roles.values()):
        raise DatasetError(f"{call_id}: speaker role mapping does not match transcript labels")
    return {
        "results": transcript,
        "speaker_roles": roles,
        "checklist": dataset["graph_checklist"],
    }
