"""Token usage capture and cost estimation."""
import json
import operator
from pathlib import Path
from typing import Annotated, TypedDict

import pytest

# The pipeline's own packages: present with the LangGraph backend, optional for a dashboard-only install.
pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send  # noqa: E402

from fyp_evaluation_engine.runner import run_evaluation
from fyp_evaluation_engine.usage import (
    NOT_RECORDED,
    invoke_with_usage,
    load_pricing,
    model_cost,
    normalize_usage,
    summarize_run_usage,
)

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
PRICING = {"currency": "USD", "models": {"gemini-x": {"input": 1.0, "output": 4.0, "cached_input": 0.25}}}


def _usage(input_tokens, output_tokens, reasoning=0, cached=0):
    return {
        "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens,
        "input_token_details": {"cache_read": cached}, "output_token_details": {"reasoning": reasoning},
    }


def _message(input_tokens, output_tokens):
    return AIMessage(content="ok", usage_metadata=_usage(input_tokens, output_tokens),
                     response_metadata={"model_name": "models/gemini-x"})


class _State(TypedDict, total=False):
    items: list[str]
    seen: Annotated[list[str], operator.add]


def _fan_out_graph(model):
    """Parallel branches that call the model without passing any config, like check_item does."""
    def route(state):
        return [Send("work", {"item": item}) for item in state["items"]]

    def work(task):
        model.invoke("judge this")
        return {"seen": [task["item"]]}

    builder = StateGraph(_State)
    builder.add_node("work", work)
    builder.add_conditional_edges(START, route, ["work"])
    builder.add_edge("work", END)
    return builder.compile()


def test_callback_totals_tokens_across_parallel_branches_without_touching_the_nodes():
    model = GenericFakeChatModel(messages=iter([_message(100, 10) for _ in range(6)]))
    output = invoke_with_usage(_fan_out_graph(model), {"items": list("abcdef")})
    assert sorted(output["seen"]) == list("abcdef")
    usage = normalize_usage(output["usage_metadata"])
    assert usage == {"gemini-x": {"input_tokens": 600, "output_tokens": 60, "total_tokens": 660,
                                  "reasoning_tokens": 0, "cache_read_tokens": 0}}


def test_tokens_spent_before_a_failure_are_still_reported():
    model = GenericFakeChatModel(messages=iter([_message(50, 5)]))

    def boom(_state):
        model.invoke("first call succeeds")
        raise RuntimeError("graph blew up")

    builder = StateGraph(_State)
    builder.add_node("boom", boom)
    builder.add_edge(START, "boom")
    builder.add_edge("boom", END)
    with pytest.raises(RuntimeError) as caught:
        invoke_with_usage(builder.compile(), {"items": []})
    assert normalize_usage(caught.value.usage_metadata)["gemini-x"]["total_tokens"] == 55


def test_cost_uses_cached_input_rate_and_counts_reasoning_inside_output():
    tokens = normalize_usage({"models/gemini-x": _usage(1_000_000, 500_000, reasoning=200_000, cached=400_000)})["gemini-x"]
    assert tokens["reasoning_tokens"] == 200_000 and tokens["output_tokens"] == 500_000
    # 600k fresh input x $1 + 400k cached x $0.25 + 500k output x $4 (reasoning is already inside output)
    assert model_cost(tokens, PRICING["models"]["gemini-x"]) == pytest.approx(0.6 + 0.1 + 2.0)


def test_summary_totals_per_call_and_prices_used():
    summary = summarize_run_usage(
        {"CALL_001": {"gemini-x": _usage(1_000_000, 0)}, "CALL_002": {"gemini-x": _usage(0, 1_000_000)}}, PRICING)
    assert summary["token_usage"]["total_tokens"] == 2_000_000
    assert summary["token_usage"]["by_call_id"]["CALL_002"]["output_tokens"] == 1_000_000
    cost = summary["estimated_cost"]
    assert cost["amount"] == pytest.approx(5.0)
    assert cost["by_call_id"] == {"CALL_001": pytest.approx(1.0), "CALL_002": pytest.approx(4.0)}
    assert cost["prices_used"]["gemini-x"]["output"] == 4.0 and cost["unpriced_models"] == []


def test_missing_prices_record_tokens_but_never_a_zero_cost():
    summary = summarize_run_usage({"CALL_001": {"gemini-x": _usage(10, 5)}}, {"models": {"gemini-x": {"input": None, "output": None}}})
    assert summary["token_usage"]["total_tokens"] == 15
    assert summary["estimated_cost"]["amount"] is None
    assert summary["estimated_cost"]["unpriced_models"] == ["gemini-x"]


def test_nothing_recorded_is_a_placeholder_not_zero():
    assert summarize_run_usage({}, PRICING) == {
        "token_usage": NOT_RECORDED, "estimated_cost": NOT_RECORDED, "trace_ids": NOT_RECORDED}


def test_runner_records_usage_in_run_metadata_and_artifacts(tmp_path):
    def invoke(value):
        chunks = [{"chunk_id": f"c{i}", "source_segment_id": s["id"]} for i, s in enumerate(value["results"]["audio_segments"])]
        findings = [{"checklist_id": item["id"], "verdict": "Pass", "spans": []} for item in value["checklist"]]
        return {"chunks": chunks, "report": {"findings": findings, "summary": {}, "warnings": []},
                "usage_metadata": {"models/gemini-3.5-flash": _usage(1000, 100)}}

    result = run_evaluation(EVALUATION_ROOT / "datasets" / "pilot", tmp_path, run_id="with-usage", graph_invoke=invoke)
    observability = result["metadata"]["observability"]
    assert observability["token_usage"]["total_tokens"] == 5 * 1100
    assert set(observability["token_usage"]["by_call_id"]) == {f"CALL_00{i}" for i in range(1, 6)}
    stored = json.loads((tmp_path / "with-usage" / "metadata.json").read_text(encoding="utf-8"))
    assert stored["observability"]["token_usage"]["input_tokens"] == 5000


def test_runs_without_usage_stay_valid_with_placeholders(tmp_path):
    def invoke(value):
        return {"chunks": [], "report": {"findings": [], "summary": {}, "warnings": []}}

    result = run_evaluation(EVALUATION_ROOT / "datasets" / "pilot", tmp_path, run_id="no-usage",
                            call_ids=["CALL_001"], graph_invoke=invoke)
    assert result["metadata"]["observability"]["token_usage"] == NOT_RECORDED


def test_versioned_model_names_match_the_configured_price():
    summary = summarize_run_usage({"CALL_001": {"models/gemini-x-001": _usage(1_000_000, 0)}}, PRICING)
    assert summary["estimated_cost"]["amount"] == pytest.approx(1.0)
    other = summarize_run_usage({"CALL_001": {"some-other-model": _usage(10, 10)}}, PRICING)
    assert other["estimated_cost"]["amount"] is None
    assert other["estimated_cost"]["unpriced_models"] == ["some-other-model"]
