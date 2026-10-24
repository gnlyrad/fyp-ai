"""Offline provider-transport and scheduling regressions. Never contacts Google."""
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import SimpleNamespace as Box

import pytest
from google.genai import types
from pydantic import BaseModel

from compliance_pipeline import llm, retrieval
from compliance_pipeline.gemini_batch import (
    BatchTransportError, NativeBatchTransport, make_request, parse_response, response_usage,
)
from compliance_pipeline.graph import graph, _route_after_intent
from compliance_pipeline.config import resolve_checklist
from compliance_pipeline.nodes.check_batch import check_batch_node
from compliance_pipeline.usage import record_native_usage, track_usage


class Answer(BaseModel):
    answer: str


def request(text="request", *, limit=1024, thinking="low"):
    return make_request("shared system", text, Answer, max_output_tokens=limit,
                        thinking_level=thinking, seed=42, temperature=1)


def response(answer="ok", *, valid=True):
    return types.GenerateContentResponse(
        model_version="gemini-fake",
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=10, thoughts_token_count=4,
            cached_content_token_count=20, total_token_count=114),
        candidates=[types.Candidate(content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(
                name="Answer", args={"answer": answer} if valid else {}))]))],
    )


class FakeBatches:
    def __init__(self, *, state="JOB_STATE_SUCCEEDED", bad_ids=False, fail_create=False):
        self.calls = []
        self.state, self.bad_ids, self.fail_create = state, bad_ids, fail_create

    def create(self, *, model, src, config):
        self.calls.append(src)
        if self.fail_create:
            raise TimeoutError("sensitive-provider-text")
        rows = [types.InlinedResponse(
            metadata={"request_id": "wrong-id"} if self.bad_ids else item.metadata,
            response=response(item.contents[0].parts[0].text)) for item in reversed(src)]
        return Box(name="batches/fake", state=self.state,
                   dest=Box(inlined_responses=rows))

    def get(self, *, name):
        return Box(name=name, state=self.state, dest=None)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(retrieval, "get_rules", lambda item: [])
    monkeypatch.setenv("COMPLIANCE_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("COMPLIANCE_LLM_MAX_ATTEMPTS", "2")
    yield
    llm.reset_backend()


@pytest.mark.parametrize("group_size", [1, 2])
def test_warmup_finishes_first_normal_request_then_retains_every_answer_once(monkeypatch, group_size):
    monkeypatch.setenv("COMPLIANCE_CHECK_CACHE_WARMUP", "1")
    monkeypatch.setenv("COMPLIANCE_CHECK_BATCH_SIZE", str(group_size))
    entered, release = threading.Event(), threading.Event()
    checked = []
    lock = threading.Lock()

    def pass_answer():
        return {"sub_criteria": [{"id": "met", "status": "PASS",
                                 "evidence": [{"segment_id": "seg_0002", "quote": "Confirmed."}]}],
                "confidence": 0.95}

    def backend(name, system, user, schema, model):
        if name in {"check_item", "check_batch"}:
            ids = [f"Q{i}" for i in range(5) if f"Requirement {i}?" in user]
            with lock:
                checked.extend(ids)
            if "Q0" in ids:
                entered.set()
                assert release.wait(timeout=5)
            if name == "check_batch":
                return {"answers": [{"item_id": item_id, "result": pass_answer()} for item_id in ids]}
            return pass_answer()
        return {}

    llm.set_backend(backend)
    value = {"call_id": "generic-warmup",
             "speaker_roles": {"spk_0": "Advisor", "spk_1": "Client"},
             "checklist": [{"id": f"Q{i}", "question": f"Requirement {i}?"} for i in range(5)],
             "results": {"audio_segments": [
                 {"speaker_label": "spk_0", "transcript": "Please confirm.", "start_time": 0, "end_time": 40},
                 {"speaker_label": "spk_1", "transcript": "Confirmed.", "start_time": 40, "end_time": 90}]}}
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(graph.invoke, value)
        try:
            assert entered.wait(timeout=5)
            assert checked == [f"Q{i}" for i in range(group_size)]
        finally:
            release.set()
        output = future.result(timeout=10)
    assert sorted(checked) == [f"Q{i}" for i in range(5)]
    assert len(output["final_results"]) == 5
    assert all(item.status == "PASS" for item in output["final_results"])
    assert not output.get("errors")


def test_warmup_adds_no_extra_call_for_a_single_group_and_is_skipped_for_native_batch(monkeypatch):
    value = {"checklist": [item.model_dump() for item in resolve_checklist([
        {"id": "Q", "question": "Requirement?"}])]}
    monkeypatch.setenv("COMPLIANCE_CHECK_CACHE_WARMUP", "1")
    assert isinstance(_route_after_intent(value), list)
    value["checklist"] *= 2
    value["checklist"][1] = {**value["checklist"][1], "id": "other"}
    monkeypatch.setenv("COMPLIANCE_CHECK_TRANSPORT", "batch")
    assert isinstance(_route_after_intent(value), list)


def test_native_batch_coalesces_independent_requests_and_matches_out_of_order_ids(tmp_path):
    api = FakeBatches()
    transport = NativeBatchTransport(Box(batches=api), tmp_path, gather_seconds=0.2, max_wait_seconds=5)
    barrier = threading.Barrier(8)
    def call(i):
        barrier.wait(timeout=3)
        row, name = transport.invoke("gemini-fake", request(str(i)))
        return parse_response(row.response, Answer).answer
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(call, range(8))) == [str(i) for i in range(8)]
        assert len(api.calls) == 1 and len(api.calls[0]) == 8
        receipt = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert receipt["job_name"] == "batches/fake"
        assert len(receipt["usage_by_request"]) == 8
        assert "shared system" not in json.dumps(receipt)
        assert not any("contents" in key for key in receipt)
    finally:
        transport.close()


@pytest.mark.parametrize("failure", ["uncertain_submission", "wrong_ids", "pending", "failed_job"])
def test_native_batch_failures_never_create_a_second_billed_job(tmp_path, failure):
    api = FakeBatches(
        state="JOB_STATE_PENDING" if failure == "pending" else
              "JOB_STATE_FAILED" if failure == "failed_job" else "JOB_STATE_SUCCEEDED",
        bad_ids=failure == "wrong_ids", fail_create=failure == "uncertain_submission")
    transport = NativeBatchTransport(Box(batches=api), tmp_path, gather_seconds=0,
                                    poll_seconds=0.01, max_wait_seconds=0.05)
    try:
        with pytest.raises(BatchTransportError):
            transport.invoke("gemini-fake", request())
        assert len(api.calls) == 1
        receipt = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert "sensitive-provider-text" not in json.dumps(receipt)
        if failure == "uncertain_submission":
            assert receipt["state"] == "submission_uncertain"
        if failure == "pending":
            assert receipt["state"] == "pending_after_wait"
            assert receipt["job_name"] == "batches/fake"
        if failure == "wrong_ids":
            assert receipt["usage_by_response"][0]["usage"]["total_tokens"] == 114
    finally:
        transport.close()


def test_uncertain_grouped_job_does_not_fall_back_to_new_individual_jobs():
    calls = []
    def backend(*args):
        calls.append(args[0])
        raise BatchTransportError("Already submitted; pending")
    llm.set_backend(backend)
    items = resolve_checklist([{"id": f"Q{i}", "question": "Requirement?"} for i in range(3)])
    value = {"checklist_items": [item.model_dump() for item in items],
             "segments": [{"id": "seg_0001", "text": "Confirmed."}]}
    output = check_batch_node(value)
    assert calls == ["check_batch"]
    assert len(output["item_results"]) == 3
    assert all(item.execution_status == "prediction_failure" for item in output["item_results"])


def test_batch_request_preserves_stage_policy_and_function_calling():
    one = request(limit=16384, thinking="medium")
    assert one.config.max_output_tokens == 16384 and one.config.seed == 42
    assert one.config.thinking_config.thinking_level.value == "MEDIUM"
    assert one.config.system_instruction == "shared system"
    assert one.config.tool_config.function_calling_config.allowed_function_names == ["Answer"]
    assert one.config.tools[0].function_declarations[0].parameters_json_schema == Answer.model_json_schema()


def test_native_malformed_repair_counts_both_billed_outputs_once(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_CHECK_TRANSPORT", "batch")
    calls = []
    class Transport:
        def invoke(self, model, req):
            calls.append(req)
            return types.InlinedResponse(response=response(valid=len(calls) > 1)), "batches/fake"
    monkeypatch.setattr(llm, "_get_native_transport", lambda: Transport())

    def node(state):
        result = llm.get_llm_response("check_item", "system", "user", Answer)
        assert result.ok and result.attempts == 2
        return {"answer": result.value.answer}
    output = track_usage(node)({})
    assert len(calls) == 2 and "Output validation failed" in calls[1].contents[0].parts[0].text
    assert output["usage"]["gemini-fake"]["total_tokens"] == 228
    assert output["usage"]["gemini-fake"]["output_tokens"] == 28
    assert output["usage_by_transport"]["batch"] == output["usage"]
    assert len(output["llm_calls"]) == 1
    assert len(output["llm_calls"][0]["native_batch_jobs"]) == 2


def test_partial_native_error_usage_is_preserved_without_retry(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_CHECK_TRANSPORT", "batch")
    class Transport:
        def invoke(self, model, req):
            return Box(response=response(), error=Box(code=500)), "batches/partial"
    monkeypatch.setattr(llm, "_get_native_transport", lambda: Transport())
    def node(state):
        result = llm.get_llm_response("check_item", "s", "u", Answer)
        assert not result.ok and result.attempts == 1 and not result.recovery_allowed
        return {}
    output = track_usage(node)({})
    assert output["usage"]["gemini-fake"]["total_tokens"] == 114


def test_native_usage_in_worker_context_is_not_lost_or_double_counted():
    from contextvars import copy_context
    def node(state):
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(copy_context().run, record_native_usage,
                                "gemini-fake", response_usage(response())) for _ in range(2)]
            for job in jobs:
                job.result()
        return {}
    output = track_usage(node)({})
    assert output["usage"]["gemini-fake"]["total_tokens"] == 228
    assert output["usage_by_transport"]["batch"]["gemini-fake"]["output_token_details"]["reasoning"] == 8


@pytest.mark.parametrize("code", [401, 402, 403, 404])
def test_permanent_provider_errors_do_not_trigger_individual_fallback(code):
    from google.genai.errors import ClientError
    calls = []
    def backend(*args):
        calls.append(args[0])
        raise ClientError(code, {"error": {"code": code, "message": "not logged"}})
    llm.set_backend(backend)
    items = resolve_checklist([{"id": f"Q{i}", "question": "Requirement?"} for i in range(2)])
    output = check_batch_node({"checklist_items": [item.model_dump() for item in items],
                              "segments": [{"id": "seg_0001", "text": "Confirmed."}]})
    assert calls == ["check_batch"]
    assert all(item.execution_status == "prediction_failure" for item in output["item_results"])
    assert all("Native batch" not in item.reasoning for item in output["item_results"])


def test_native_usage_missing_is_unknown_not_free(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_CHECK_TRANSPORT", "batch")
    class Transport:
        def invoke(self, model, req):
            result = response().model_copy(update={"usage_metadata": None})
            return types.InlinedResponse(response=result), "batches/no-usage"
    monkeypatch.setattr(llm, "_get_native_transport", lambda: Transport())
    def node(state):
        result = llm.get_llm_response("check_item", "s", "u", Answer)
        assert result.ok
        return {}
    output = track_usage(node)({})
    assert "usage" not in output
    assert output["llm_calls"][0]["native_batch_jobs"][0]["usage_reported"] is False


def test_standard_and_native_tokens_are_merged_once_and_keep_pricing_tiers_separate():
    import itertools
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
    message = AIMessage(content="ok", usage_metadata={
        "input_tokens": 50, "output_tokens": 5, "total_tokens": 55},
        response_metadata={"model_name": "gemini-fake"})
    model = GenericFakeChatModel(messages=itertools.cycle([message]))
    def node(state):
        model.invoke("online")
        record_native_usage("gemini-fake", response_usage(response()))
        return {}
    output = track_usage(node)({})
    assert output["usage"]["gemini-fake"]["total_tokens"] == 169
    assert output["usage_by_transport"]["standard"]["gemini-fake"]["total_tokens"] == 55
    assert output["usage_by_transport"]["batch"]["gemini-fake"]["total_tokens"] == 114


def test_native_client_disables_sdk_retry_for_non_idempotent_submissions(monkeypatch):
    from unittest.mock import patch
    from langchain_backend import models
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-test-key")
    with patch("google.genai.Client") as client:
        models.create_batch_client(30)
    options = client.call_args.kwargs
    assert options["vertexai"] is False
    assert options["http_options"].timeout == 30000
    assert options["http_options"].retry_options.attempts == 1


def test_cost_report_uses_explicit_tier_rates_and_does_not_charge_thinking_twice():
    from compliance_pipeline.usage import estimate_reported_cost
    price_book = {"models": {"gemini-fake": {
        "standard": {"input": 0.75, "output": 3.75, "cached_input": 0.075},
        "batch": {"input": 0.375, "output": 1.875, "cached_input": 0.0375}}}}
    tokens = response_usage(response())
    result = estimate_reported_cost(
        {"standard": {"gemini-fake": tokens}, "batch": {"gemini-fake": tokens}}, price_book)
    assert result["by_transport"]["standard"] == pytest.approx(0.000114)
    assert result["by_transport"]["batch"] == pytest.approx(0.000057)
    assert result["amount"] == pytest.approx(0.000171)


@pytest.mark.parametrize("prices", [
    {},
    {"models": {"gemini-fake": {"standard": {"input": 0.75, "output": 3.75}}}},
    {"models": {"gemini-fake": {"batch": {"input": 0.375, "output": 1.875}}}},
])
def test_missing_transport_or_cache_prices_are_not_treated_as_free(prices):
    from compliance_pipeline.usage import estimate_reported_cost
    result = estimate_reported_cost({"batch": {"gemini-fake": response_usage(response())}}, prices)
    assert result["amount"] is None and result["unpriced_models"] == ["batch/gemini-fake"]


def test_absent_usage_cannot_produce_a_zero_cost_claim():
    from compliance_pipeline.usage import estimate_reported_cost
    assert estimate_reported_cost({}, {"models": {}})["amount"] is None


def test_partially_successful_job_preserves_good_rows_and_reports_failed_ids(tmp_path):
    class Partial(FakeBatches):
        def create(self, **kwargs):
            job = super().create(**kwargs)
            row = job.dest.inlined_responses[0]
            job.dest.inlined_responses[0] = Box(
                metadata=row.metadata, response=None, error=Box(code=13))
            return job
    api = Partial(state="JOB_STATE_PARTIALLY_SUCCEEDED")
    transport = NativeBatchTransport(Box(batches=api), tmp_path, gather_seconds=0.2, max_wait_seconds=3)
    barrier = threading.Barrier(2)
    def call(i):
        barrier.wait(timeout=2)
        return transport.invoke("gemini-fake", request(str(i)))[0]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(call, range(2)))
        assert sum(row.response is not None for row in rows) == 1
        receipt = json.loads(next(tmp_path.glob("*.json")).read_text())
        assert receipt["state"] == "JOB_STATE_PARTIALLY_SUCCEEDED"
        assert len(receipt["failed_request_ids"]) == 1 and len(api.calls) == 1
    finally:
        transport.close()


def test_pending_job_name_is_retained_in_pipeline_diagnostics(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_CHECK_TRANSPORT", "batch")
    class Transport:
        def invoke(self, model, req):
            raise BatchTransportError("Still pending", "batches/existing")
    monkeypatch.setattr(llm, "_get_native_transport", lambda: Transport())
    def node(state):
        result = llm.get_llm_response("check_item", "s", "u", Answer)
        assert not result.ok and result.attempts == 1
        return {}
    output = track_usage(node)({})
    entry = output["llm_calls"][0]["native_batch_jobs"][0]
    assert entry["job_name"] == "batches/existing" and entry["usage_reported"] is False


def test_empty_model_group_is_not_recorded_as_free_usage():
    from compliance_pipeline.usage import estimate_reported_cost
    assert estimate_reported_cost({"batch": {}}, {})["amount"] is None


def test_unused_cache_price_can_be_unknown_without_breaking_fresh_token_costs():
    from compliance_pipeline.usage import estimate_reported_cost
    price = {"models": {"gemini-fake": {"standard": {
        "input": 0.75, "output": 3.75, "cached_input": None}}}}
    tokens = response_usage(response())
    tokens["input_token_details"] = {"cache_read": 0}
    result = estimate_reported_cost({"standard": {"gemini-fake": tokens}}, price)
    assert result["amount"] == pytest.approx(0.0001275)
