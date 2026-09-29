"""Regression checks for merging the staged graph with the Gemini evaluator."""
from pathlib import Path

import pytest
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel

from compliance_pipeline import llm, retrieval
from compliance_pipeline.graph import graph
from compliance_pipeline.nodes.ingest import ingest_node
from compliance_pipeline.nodes.check_item import check_item_node
from compliance_pipeline.nodes.verify import verify_node
from compliance_pipeline.schemas import ItemResult, SubCriterionResult, Status
from fyp_evaluation_engine.current_pipeline import prepare_input, adapt_output
from fyp_evaluation_engine.runner import run_evaluation


@pytest.fixture(autouse=True)
def isolated_backend(monkeypatch):
    monkeypatch.setattr(llm, "TRANSIENT_BACKOFF_S", [0, 0])
    monkeypatch.delenv("COMPLIANCE_LLM_MAX_ATTEMPTS", raising=False)
    monkeypatch.setattr(retrieval, "get_rules", lambda item: [])
    yield
    llm.reset_backend()


class Answer(BaseModel):
    answer: str


@pytest.mark.parametrize("code,attempts", [(403, 1), (404, 1), (429, 3), (503, 3)])
def test_provider_failures_are_bounded_and_sanitized(code, attempts):
    def fail(*args):
        cls = ClientError if code < 500 else ServerError
        raise cls(code, {"error": {"code": code, "message": "secret-input"}})
    llm.set_backend(fail)
    result = llm.get_llm_response("test", "", "", Answer)
    assert not result.ok and result.attempts == attempts
    assert str(code) in result.error and "secret-input" not in result.error


def test_malformed_response_retried_once():
    llm.set_backend(lambda *args: {})
    result = llm.get_llm_response("test", "", "", Answer)
    assert not result.ok and result.attempts == 2


def test_llm_client_uses_shared_gemini_configuration(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_clients", {})
    monkeypatch.setattr(llm, "create_chat_model", lambda **kwargs: calls.append(kwargs) or object())
    llm._get_client("test-model")
    assert calls[0]["model_id"] == "test-model"
    assert calls[0]["max_retries"] == 0


def sample():
    return {"checklist": [{"id": "C02", "question": "Was the request confirmed?"}],
            "results": {"audio_segments": [
                {"id": 91, "speaker_label": "spk_0", "transcript": "First."},
                {"id": 7, "speaker_label": "spk_0", "transcript": "Second."},
                {"id": 33, "speaker_label": "spk_1", "transcript": "Confirmed."}]}}


def test_namespacing_keeps_pilot_c02_as_a_stage2_question():
    value = prepare_input(sample())
    assert value["checklist"] == [{"id": "eval_C02", "question": "Was the request confirmed?"}]
    state = ingest_node(value)
    item = next(i for i in state["checklist"] if i["id"] == "eval_C02")
    assert item["stage"] == 2
    assert any(i["id"] == "C02" and i["stage"] == 1 for i in state["checklist"])


@pytest.mark.parametrize("start,end,expected", [(7, 14, [7]), (0, 14, [91, 7])])
def test_evidence_maps_merged_turns_to_original_ids(start, end, expected):
    value = sample()
    state = ingest_node(prepare_input(value))
    state["final_results"] = [ItemResult(item_id="eval_C02", status=Status.PASS,
        evidence=[{"segment_id": "seg_0001", "start_char": start, "end_char": end}])]
    bridged = adapt_output(state, value["results"])
    assert [c["source_segment_id"] for c in bridged["chunks"]] == expected


def test_invalid_evidence_is_not_silently_dropped():
    value = sample()
    state = ingest_node(prepare_input(value))
    state["final_results"] = [ItemResult(item_id="eval_C02", status=Status.PASS,
        evidence=[{"segment_id": "seg_0091", "start_char": 0, "end_char": 5}])]
    bridged = adapt_output(state, value["results"])
    assert bridged["report"]["findings"][0]["spans"]
    assert not bridged["chunks"]


def test_check_failure_differs_from_semantic_unclear():
    state = ingest_node(prepare_input(sample()))
    item = next(i for i in state["checklist"] if i["id"] == "eval_C02")
    task = {**state, "item_id": item["id"], "checklist_item": item}
    llm.set_backend(lambda *args: {"sub_criteria": [{"id": "met", "status": "CANNOT_DETERMINE"}]})
    assert check_item_node(task)["item_results"][0].execution_status == "success"
    llm.set_backend(lambda *args: {})
    result = check_item_node(task)["item_results"][0]
    assert result.execution_status == "prediction_failure"
    assert result.failure_category == "incomplete_structured_output"


def test_failed_verification_is_propagated():
    state = ingest_node(prepare_input(sample()))
    state["item_results"] = [ItemResult(item_id="eval_C02", status=Status.FAIL,
        sub_criteria=[SubCriterionResult(id="met", status=Status.FAIL)])]
    def fail(*args):
        raise ClientError(403, {"error": {"code": 403, "message": "denied"}})
    llm.set_backend(fail)
    result = verify_node(state)["verified_results"][0]
    assert result.execution_status == "prediction_failure" and not result.verified


def test_full_graph_to_evaluation_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPLIANCE_OUTPUT_DIR", str(tmp_path / "graph"))
    def backend(name, system, user, schema, model):
        if name == "check_item":
            return schema.model_validate({"sub_criteria": [{"id": "met", "status": "CANNOT_DETERMINE"}],
                                          "confidence": 0.9, "reasoning": "Insufficient evidence."})
        return schema.model_validate({})
    llm.set_backend(backend)
    dataset = Path(__file__).resolve().parents[2] / "evaluation" / "datasets" / "pilot"
    result = run_evaluation(dataset, tmp_path / "runs", call_ids=["CALL_001"])
    assert result["metadata"]["call_executions"][0]["status"] == "success"
    assert result["summary"]["overall"]["compliance"]["outcome_population"]["prediction_failures"] == 0
    assert result["summary"]["overall"]["compliance"]["outcome_population"]["abstentions"] == 6
    assert (tmp_path / "graph" / "CALL_001.json").exists()


def test_display_brackets_resolve_only_existing_segment_ids():
    from compliance_pipeline.nodes._common import resolve_evidence
    from compliance_pipeline.schemas import EvidenceQuote
    segments = [{"id": "seg_0001", "text": "Confirmed."}]
    refs, errors = resolve_evidence(segments, [EvidenceQuote(segment_id="[seg_0001]", quote="Confirmed.")])
    assert not errors and refs[0].segment_id == "seg_0001" and refs[0].verified
    refs, errors = resolve_evidence(segments, [EvidenceQuote(segment_id="[seg_9999]", quote="Confirmed.")])
    assert not refs and errors
