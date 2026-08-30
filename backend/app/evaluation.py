from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .data import load_calls
from .rag import ComplianceRAGWorkflow, analyse_checklist_item, build_chunks

ARTIFACTS_DIR = Path(os.getenv("EVALUATION_OUTPUT_DIR", Path(__file__).resolve().parents[1] / "artifacts"))


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def macro_f1(expected: list[str], predicted: list[str]) -> float:
    """Macro F1 over the evaluation labels, including labels with no predictions."""
    labels = sorted(set(expected) | set(predicted))
    scores = []
    for label in labels:
        true_positive = sum(actual == label and guess == label for actual, guess in zip(expected, predicted))
        false_positive = sum(actual != label and guess == label for actual, guess in zip(expected, predicted))
        false_negative = sum(actual == label and guess != label for actual, guess in zip(expected, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    return round(sum(scores) / len(scores), 4) if scores else 0


def run_evaluation(log_to_mlflow: bool = False) -> dict:
    calls = load_calls()
    total = correct = evidence_supported = retrieval_hits = 0
    precision_at_k_total = 0.0
    expected_verdicts: list[str] = []
    predicted_verdicts: list[str] = []
    predictions = []
    workflow = ComplianceRAGWorkflow()
    started_at = time.perf_counter()
    for call in calls:
        chunks = build_chunks(call["utterances"])
        for item in call["checklist"]:
            item_started_at = time.perf_counter()
            prediction = analyse_checklist_item(item, chunks, workflow)
            prediction["latency_seconds"] = round(time.perf_counter() - item_started_at, 3)
            total += 1
            correct += prediction["verdict"] == item["expected_verdict"]
            evidence_supported += bool(prediction["evidence_quote"])
            expected_verdicts.append(item["expected_verdict"])
            predicted_verdicts.append(prediction["verdict"])
            expected_quote = item.get("evidence_quote", "").lower()
            relevant_chunks = [
                chunk for chunk in prediction["retrieved_evidence"]
                if expected_quote and expected_quote in chunk["text"].lower()
            ]
            retrieval_hits += bool(relevant_chunks)
            precision_at_k_total += len(relevant_chunks) / max(1, len(prediction["retrieved_evidence"]))
            predictions.append({"call_id": call["call_id"], "expected": item["expected_verdict"], **prediction})

    metrics = {
        "verdict_accuracy": round(correct / total, 4) if total else 0,
        "macro_f1": macro_f1(expected_verdicts, predicted_verdicts),
        "evidence_support_rate": round(evidence_supported / total, 4) if total else 0,
        "retrieval_recall_at_k": round(retrieval_hits / total, 4) if total else 0,
        "retrieval_precision_at_k": round(precision_at_k_total / total, 4) if total else 0,
        "average_latency_seconds": round((time.perf_counter() - started_at) / total, 3) if total else 0,
        # Azure token billing is deployment-specific; populate this once pricing
        # telemetry is connected. Local Ollama runs have no per-request charge.
        "estimated_cost_usd": 0.0,
        "total_checklist_items": total,
        "calls_evaluated": len(calls),
    }
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "retrieval_method": workflow.settings.retrieval_provider,
            "generation_provider": workflow.settings.llm_provider,
            "model": workflow.settings.ollama_model if workflow.settings.llm_provider == "ollama" else workflow.settings.azure_openai_chat_deployment,
            "temperature": 0,
            "top_k": 3,
            "dataset_version": "synthetic-v1",
            "git_commit": git_sha(),
        },
        "metrics": metrics,
        "predictions": predictions,
    }
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    artifact = ARTIFACTS_DIR / "latest_evaluation.json"
    artifact.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if log_to_mlflow:
        try:
            import mlflow
            if os.getenv("MLFLOW_TRACKING_URI"):
                mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "clearcall-compliance-evaluation"))
            with mlflow.start_run():
                mlflow.log_params(payload["parameters"])
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(str(artifact))
        except ImportError:
            pass
    return payload
