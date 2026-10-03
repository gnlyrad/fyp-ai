"""The pipeline reports model token usage in its own output."""
import itertools
import json
import threading
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from compliance_pipeline import llm, retrieval
from compliance_pipeline.usage import merge_usage, track_usage
from fyp_evaluation_engine.runner import run_evaluation

PER_CALL = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}
MODEL = "gemini-fake"


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(llm, "TRANSIENT_BACKOFF_S", [0, 0])
    monkeypatch.delenv("COMPLIANCE_LLM_MAX_ATTEMPTS", raising=False)
    monkeypatch.setattr(retrieval, "get_rules", lambda item: [])
    yield
    llm.reset_backend()


def _model():
    message = AIMessage(content="ok", usage_metadata=PER_CALL, response_metadata={"model_name": MODEL})
    return GenericFakeChatModel(messages=itertools.cycle([message]))


def test_merge_usage_sums_per_model_and_keeps_models_separate():
    first = {"a": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}}
    second = {"a": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
              "b": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10}}
    merged = merge_usage(first, second)
    assert merged["a"]["total_tokens"] == 33 and merged["b"]["total_tokens"] == 10
    assert merge_usage(None, second) == second and merge_usage(first, None) == first


def test_track_usage_leaves_a_node_untouched_when_it_calls_no_model():
    node = track_usage(lambda state: {"x": 1})
    assert node({}) == {"x": 1}


def test_full_graph_output_carries_usage_that_the_evaluation_records(tmp_path, monkeypatch):
    monkeypatch.setenv("COMPLIANCE_OUTPUT_DIR", str(tmp_path / "graph"))
    model, calls, lock = _model(), [0], threading.Lock()

    def backend(name, system, user, schema, model_id):
        model.invoke("billed call")  # a real LangChain call, so the usage callback fires
        with lock:
            calls[0] += 1
        if name == "check_item":
            return schema.model_validate({"sub_criteria": [{"id": "met", "status": "CANNOT_DETERMINE"}],
                                          "confidence": 0.9, "reasoning": "Insufficient evidence."})
        return schema.model_validate({})

    llm.set_backend(backend)
    dataset = Path(__file__).resolve().parents[2] / "evaluation" / "datasets" / "pilot"
    result = run_evaluation(dataset, tmp_path / "runs", call_ids=["CALL_001"])

    assert calls[0] >= 10  # at least one model call per checklist question
    expected_input = calls[0] * PER_CALL["input_tokens"]

    # 1. The pipeline's own output file includes the usage, tokens only (no price).
    payload = json.loads((tmp_path / "graph" / "CALL_001.json").read_text(encoding="utf-8"))
    assert payload["usage"][MODEL]["input_tokens"] == expected_input
    assert not any("cost" in key or "price" in key for key in payload["usage"][MODEL])

    # 2. The raw graph output saved with the run has it too.
    raw = json.loads((tmp_path / "runs" / result["metadata"]["run_id"] / "raw" / "CALL_001.json").read_text(encoding="utf-8"))
    assert raw["usage"][MODEL]["input_tokens"] == expected_input

    # 3. The evaluation records the same figure, not a double count.
    recorded = result["metadata"]["observability"]["token_usage"]
    assert recorded["input_tokens"] == expected_input
    assert recorded["by_model"][MODEL]["output_tokens"] == calls[0] * PER_CALL["output_tokens"]


def test_only_the_harmless_defs_warning_is_silenced(caplog):
    import logging

    logger = logging.getLogger("langchain_google_genai._function_utils")
    with caplog.at_level(logging.WARNING, logger=logger.name):
        logger.warning("Key '$defs' is not supported in schema, ignoring")
        logger.warning("Key 'patternProperties' is not supported in schema, ignoring")
    messages = [record.getMessage() for record in caplog.records]
    assert messages == ["Key 'patternProperties' is not supported in schema, ignoring"]
