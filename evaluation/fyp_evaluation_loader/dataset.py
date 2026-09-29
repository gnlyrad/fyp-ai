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
QUESTION_TYPES = {"COMPLIANCE", "CLIENT_KYC", "TRANSACTION"}
# Checklist type is the source category; the evaluation family says HOW a question is scored.
# It matches the ground-truth `expected.kind` of every check, which the loader enforces.
EVALUATION_FAMILY_BY_QUESTION_TYPE = {
    "COMPLIANCE": "compliance",
    "CLIENT_KYC": "extraction",
    "TRANSACTION": "extraction",
}


class DatasetError(ValueError):
    """The published dataset is missing, malformed, or internally inconsistent."""


def evaluation_family_for_question_type(question_type: str) -> str:
    """Return ``compliance`` or ``extraction`` for a checklist question type."""
    try:
        return EVALUATION_FAMILY_BY_QUESTION_TYPE[question_type]
    except KeyError:
        raise DatasetError(f"Unsupported question type: {question_type!r}") from None


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
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("question"), str):
            raise DatasetError(f"checklist[{index}] needs string id and question")
        if not item["id"] or not item["question"].strip() or item["id"] in seen:
            raise DatasetError(f"checklist[{index}] has an empty or duplicate id/question")
        seen.add(item["id"])
        projected.append({"id": item["id"], "question": item["question"]})
    return projected


def _validate_checklist(checklist: Any) -> None:
    _project_checklist(checklist)
    for index, item in enumerate(checklist):
        if item.get("type") not in QUESTION_TYPES:
            raise DatasetError(f"checklist[{index}] has an unsupported type")
        if item["type"] == "COMPLIANCE":
            if item.get("requirement_type") not in {"OBLIGATION", "PROHIBITION"}:
                raise DatasetError(f"checklist[{index}] needs a compliance requirement_type")
        elif not isinstance(item.get("field"), str) or not item["field"].strip():
            raise DatasetError(f"checklist[{index}] needs an extraction field")


def _validate_call_documents(
    call_id: str,
    checklist: list[dict[str, Any]],
    ground_truth: Any,
    transcript: Any,
    roles: Any,
) -> None:
    if not isinstance(ground_truth, dict) or ground_truth.get("call_id") != call_id:
        raise DatasetError(f"{call_id}: ground_truth call_id does not match directory")
    checks = ground_truth.get("checks")
    if not isinstance(checks, list):
        raise DatasetError(f"{call_id}: ground_truth requires checks")
    checklist_by_id = {item["id"]: item for item in checklist}
    check_ids = [check.get("question_id") for check in checks if isinstance(check, dict)]
    if len(check_ids) != len(checks) or set(check_ids) != set(checklist_by_id) or len(check_ids) != len(set(check_ids)):
        raise DatasetError(f"{call_id}: ground_truth checks must match checklist IDs exactly")
    if not isinstance(transcript, dict) or not isinstance(transcript.get("audio_segments"), list):
        raise DatasetError(f"{call_id}: transcript requires audio_segments")
    segments = transcript["audio_segments"]
    segment_ids = {segment.get("id") for segment in segments if isinstance(segment, dict)}
    if len(segment_ids) != len(segments) or None in segment_ids:
        raise DatasetError(f"{call_id}: transcript segment IDs must be present and unique")
    if not isinstance(roles, dict) or not roles:
        raise DatasetError(f"{call_id}: speaker_roles.json requires a non-empty role mapping")
    labels = {segment.get("speaker_label") for segment in segments if isinstance(segment, dict)}
    if set(roles) != labels or any(role not in {"RM", "Customer"} for role in roles.values()):
        raise DatasetError(f"{call_id}: speaker role mapping does not match transcript labels")
    for check in checks:
        question_id = check["question_id"]
        expected = check.get("expected")
        turns = check.get("gold_evidence_turns")
        if not isinstance(expected, dict) or not isinstance(turns, list):
            raise DatasetError(f"{call_id}/{question_id}: expected and gold_evidence_turns are required")
        if any(turn not in segment_ids for turn in turns):
            raise DatasetError(f"{call_id}/{question_id}: gold evidence references an unknown turn")
        family = evaluation_family_for_question_type(checklist_by_id[question_id]["type"])
        if expected.get("kind") != family:
            raise DatasetError(f"{call_id}/{question_id}: expected.kind does not match the {family} evaluation family")
        if family == "compliance":
            if expected.get("expected_verdict") not in {"PASS", "FAIL"}:
                raise DatasetError(f"{call_id}/{question_id}: invalid compliance expectation")
        elif "expected_value" not in expected:
            raise DatasetError(f"{call_id}/{question_id}: invalid extraction expectation")


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
    _validate_checklist(checklist)
    for call_id in call_ids:
        call_dir = root / "calls" / call_id
        _validate_call_documents(
            call_id,
            checklist,
            _read_json(call_dir / "ground_truth.json"),
            _read_json(call_dir / "transcript.json"),
            _read_json(call_dir / "speaker_roles.json"),
        )
    return {
        "root": root,
        "manifest": manifest,
        "manifest_sha256": _hash(manifest),
        "checklist": [
            {**item, "evaluation_family": evaluation_family_for_question_type(item["type"])}
            for item in checklist
        ],
        "graph_checklist": _project_checklist(checklist),
    }


def call_documents(dataset: dict[str, Any], call_id: str) -> dict[str, Any]:
    """Load one already-validated call's evaluation documents."""
    if call_id not in dataset["manifest"]["call_ids"]:
        raise DatasetError(f"Call is not in this dataset: {call_id}")
    call_dir = dataset["root"] / "calls" / call_id
    documents = {
        "ground_truth": _read_json(call_dir / "ground_truth.json"),
        "transcript": _read_json(call_dir / "transcript.json"),
        "speaker_roles": _read_json(call_dir / "speaker_roles.json"),
    }
    _validate_call_documents(
        call_id,
        dataset["checklist"],
        documents["ground_truth"],
        documents["transcript"],
        documents["speaker_roles"],
    )
    return documents


def graph_input_for_call(dataset: dict[str, Any], call_id: str) -> dict[str, Any]:
    """Prepare exactly the direct input consumed by `compliance.graph.invoke`."""
    documents = call_documents(dataset, call_id)
    return {
        "results": documents["transcript"],
        "speaker_roles": documents["speaker_roles"],
        "checklist": dataset["graph_checklist"],
    }
