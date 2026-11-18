"""Speech-to-text (STT) results for the dashboard: read-only views over bundles exported by fyp-be/evaluation-engine.

A bundle lives in `evaluation/stt_runs/<run id>/` (run.json, calls/CALL_xxx.json, errors.json, confusions.json, unnormalised.json). It is
self-contained: everything but audio playback works without fyp-be. Audio is referenced by dataset / call / render ids plus its sha256 and is
resolved under the audio folder (by default the fyp-be `evaluation-dataset/datasets` folder beside fyp-ai; `STT_AUDIO_ROOT` overrides it),
only inside that folder and only when the file hash matches the recording that was evaluated.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,80}")
CALL_ID = re.compile(r"CALL_\d{3,}")
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
LOW_CONFIDENCE_LIST = 0.9  # the confidence drill-down lists words below this; the page lets you narrow it (below 0.5, 0.7, 0.9)


class SttNotFound(LookupError):
    pass


class AudioUnavailable(RuntimeError):
    """The recording cannot be served; `status` is the HTTP code and the message is safe to show."""

    def __init__(self, message: str, status: int = 404):
        super().__init__(message)
        self.status = status


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dataset_label(dataset_id: str) -> str:
    suffix = "__environment_v2"
    noisy = dataset_id.endswith(suffix)
    base = (dataset_id[:-len(suffix)] if noisy else dataset_id).replace("_", " ").title()
    return f"{base} (noisy and channel-degraded)" if noisy else f"{base} (clean)"


class SttStore:
    def __init__(self, root: str | Path, audio_root: str | Path | None = None, audio_setting: str | None = None):
        self.root = Path(root)
        self.audio_root = Path(audio_root).resolve() if audio_root else None
        self.audio_setting = audio_setting or (str(audio_root) if audio_root else None)  # as configured, for messages
        self._hashes: dict[tuple[str, int, int], str] = {}

    # ------------------------------------------------------------------ bundles

    def _bundle(self, run_id: str) -> Path:
        if not RUN_ID.fullmatch(run_id) or not (self.root / run_id / "run.json").is_file():
            raise SttNotFound(f"No speech-to-text run called {run_id!r}")
        return self.root / run_id

    def _run_summary(self, run: dict[str, Any]) -> dict[str, Any]:
        summary = run["summary"]
        return {
            "run_id": run["run_id"], "dataset_id": run["dataset_id"], "render_id": run.get("render_id"),
            "collected_at": run.get("collected_at"), "scored_at": run.get("scored_at"),
            "calls_scored": summary["calls_scored"], "dataset_call_count": len(run.get("dataset_call_ids") or []),
            "normalised_wer": summary["normalised_wer"], "entity_wer": summary["entity_wer"],
            "average_confidence": summary["average_confidence"], "speaker_attribution": summary["speaker_attribution"],
            "normalizer_version": run["versions"]["normalizer"],
            "vocabulary_name": (run.get("transcribe_job") or {}).get("vocabulary_name"),
            "language_model_name": (run.get("transcribe_job") or {}).get("language_model_name"),
        }

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for path in sorted(self.root.glob("*/run.json")) if self.root.is_dir() else []:
            try:
                runs.append(self._run_summary(_read(path)))
            except (KeyError, ValueError, OSError):
                continue  # a broken bundle must not hide the others
        return sorted(runs, key=lambda run: run.get("collected_at") or "", reverse=True)

    def list_datasets(self) -> list[dict[str, Any]]:
        by_dataset: dict[str, list[dict[str, Any]]] = {}
        for run in self.list_runs():
            by_dataset.setdefault(run["dataset_id"], []).append(run)
        return [{"dataset_id": dataset_id, "label": dataset_label(dataset_id), "run_count": len(runs),
                 "latest_run_at": runs[0].get("collected_at"), "dataset_call_count": runs[0]["dataset_call_count"]}
                for dataset_id, runs in sorted(by_dataset.items())]

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        runs = [run for run in self.list_runs() if run["dataset_id"] == dataset_id]
        if not runs:
            raise SttNotFound(f"No speech-to-text runs for dataset {dataset_id!r}")
        return {"dataset": {"dataset_id": dataset_id, "label": dataset_label(dataset_id), "dataset_call_count": runs[0]["dataset_call_count"]},
                "runs": runs}

    def run(self, run_id: str) -> dict[str, Any]:
        bundle = self._bundle(run_id)
        run = _read(bundle / "run.json")
        return {**run, "summary_view": self._run_summary(run), "errors": _read(bundle / "errors.json") if (bundle / "errors.json").is_file() else [],
                "unnormalised": _read(bundle / "unnormalised.json") if (bundle / "unnormalised.json").is_file() else {},
                "changes": _read(bundle / "changes_from_1.2.0.json") if (bundle / "changes_from_1.2.0.json").is_file() else None,
                "audio_configured": self.audio_root is not None}

    def call(self, run_id: str, call_id: str) -> dict[str, Any]:
        bundle = self._bundle(run_id)
        if not CALL_ID.fullmatch(call_id) or not (bundle / "calls" / f"{call_id}.json").is_file():
            raise SttNotFound(f"No call {call_id!r} in run {run_id!r}")
        call = _read(bundle / "calls" / f"{call_id}.json")
        run = _read(bundle / "run.json")
        order = [row["call_id"] for row in run["calls"]]
        position = order.index(call_id) if call_id in order else -1
        try:
            self._audio_file(call, check_hash=True)  # the real reason up front, including a changed recording (hash cached)
            audio = {"available": True, "reason": None}
        except AudioUnavailable as error:
            audio = {"available": False, "reason": str(error)}
        return {**call, "audio_status": audio, "run_id": run_id, "dataset_id": run["dataset_id"],
                "previous_call": order[position - 1] if position > 0 else None,
                "next_call": order[position + 1] if 0 <= position < len(order) - 1 else None,
                "run": self._run_summary(run), "normalizer_version": run["versions"]["normalizer"]}

    def run_words(self, run_id: str) -> dict[str, Any]:
        """Run-wide word lists for the confidence and speaker drill-downs, read from the exported call files (nothing is re-scored).

        Confidence buckets count each Transcribe item once per call; an item is "matched" when it is eligible (it matched the reference).
        `low_confidence` lists every display unit whose lowest confidence is below LOW_CONFIDENCE_LIST (the page filters further), the
        speaker list holds the units flagged `speaker_mismatch`, and `entity_spans` lists every annotated span with the unit to open.
        """
        bundle = self._bundle(run_id)
        run = _read(bundle / "run.json")
        edges = [0.0, 0.5, 0.7, 0.9, 1.0]
        buckets = [{"from": low, "to": high, "matched": 0, "other": 0} for low, high in zip(edges, edges[1:])]
        low_confidence: list[dict[str, Any]] = []
        speaker_mismatch: list[dict[str, Any]] = []
        entity_spans: list[dict[str, Any]] = []
        for row in run["calls"]:
            path = bundle / "calls" / f"{row['call_id']}.json"
            if not CALL_ID.fullmatch(row["call_id"]) or not path.is_file():
                continue
            seen: set[int] = set()
            call = _read(path)
            for span in call.get("entity_spans") or []:
                unit_id = next((u["id"] for u in call["units"] if u["turn_id"] == span["turn_id"] and span["group"] in u.get("entity_groups", [])), None)
                entity_spans.append({"call_id": row["call_id"], "unit": unit_id, **{key: span.get(key) for key in
                                     ("turn_id", "group", "text", "errors", "tokens", "hypothesis_tokens")}})
            for unit in call["units"]:
                for item in unit["items"]:
                    if item["item"] in seen or item.get("confidence") is None:
                        continue
                    seen.add(item["item"])
                    bucket = next(b for b in buckets if item["confidence"] < b["to"] or b["to"] == 1.0)
                    bucket["matched" if item.get("eligible") else "other"] += 1
                lowest = unit.get("min_confidence")
                is_low = lowest is not None and lowest < LOW_CONFIDENCE_LIST
                if not (is_low or unit.get("speaker_mismatch")):
                    continue
                entry = {"call_id": row["call_id"], "unit": unit["id"], "turn_id": unit["turn_id"], "ref_text": unit["ref_text"],
                         "hyp_text": unit["hyp_text"], "errors": unit["errors"], "ref_role": unit.get("ref_role"),
                         "min_confidence": unit.get("min_confidence"), "start": unit.get("start"),
                         "items": [{key: item.get(key) for key in ("content", "confidence", "speaker", "mapped_role")} for item in unit["items"]]}
                if is_low:
                    low_confidence.append(entry)
                if unit.get("speaker_mismatch"):
                    speaker_mismatch.append(entry)
        low_confidence.sort(key=lambda entry: entry["min_confidence"] if entry["min_confidence"] is not None else 1.0)
        entity_spans.sort(key=lambda span: (-(span["errors"] or 0), span["group"], span["call_id"], span["turn_id"]))
        return {"confidence_buckets": buckets, "low_confidence": low_confidence, "speaker_mismatch": speaker_mismatch, "entity_spans": entity_spans}

    # -------------------------------------------------------------------- audio

    def _sha256(self, path: Path) -> str:
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in self._hashes:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            self._hashes[key] = digest.hexdigest()
        return self._hashes[key]

    def _audio_file(self, call: dict[str, Any], *, check_hash: bool) -> Path:
        reference = call.get("audio") or {}
        if self.audio_root is None:
            raise AudioUnavailable("Audio unavailable on this machine: playback is turned off (no audio folder)")
        if not self.audio_root.is_dir():
            raise AudioUnavailable(f"Audio unavailable on this machine: the audio folder {self.audio_setting} does not exist")
        dataset, render, call_id = reference.get("dataset_id", ""), reference.get("render_id") or "", reference.get("call_id", "")
        if not (SAFE_ID.fullmatch(dataset) and SAFE_ID.fullmatch(render) and CALL_ID.fullmatch(call_id)):
            raise AudioUnavailable("Audio unavailable on this machine: the bundle's audio reference is not valid")
        path = (self.audio_root / dataset / "calls" / call_id / "renders" / render / "audio.mp3").resolve()
        if self.audio_root not in path.parents:
            raise AudioUnavailable("Audio unavailable on this machine: the reference points outside the audio folder")
        if not path.is_file():
            raise AudioUnavailable(f"Audio unavailable on this machine: no recording at {dataset}/calls/{call_id}/renders/{render}/audio.mp3 "
                                   f"under {self.audio_setting}")
        if check_hash and reference.get("audio_sha256") and self._sha256(path) != reference["audio_sha256"]:
            raise AudioUnavailable("Audio unavailable: the file on this machine is not the recording that was evaluated", status=409)
        return path

    def audio_path(self, run_id: str, call_id: str) -> Path:
        bundle = self._bundle(run_id)
        if not CALL_ID.fullmatch(call_id) or not (bundle / "calls" / f"{call_id}.json").is_file():
            raise SttNotFound(f"No call {call_id!r} in run {run_id!r}")
        return self._audio_file(_read(bundle / "calls" / f"{call_id}.json"), check_hash=True)
