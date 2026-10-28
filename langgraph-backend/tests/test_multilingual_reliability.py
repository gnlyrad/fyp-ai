"""NLP-only synthetic regressions. No benchmark labels or live provider calls."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from compliance_pipeline import llm, retrieval
from compliance_pipeline.config import resolve_checklist
from compliance_pipeline.grouped_output import recover_grouped_output
from compliance_pipeline.nodes._common import _find_normalised, resolve_evidence
from compliance_pipeline.nodes.check_batch import check_batch_node
from compliance_pipeline.nodes.check_item import check_item_node
from compliance_pipeline.nodes.quality_check import quality_check_node
from compliance_pipeline.nodes.verify import verify_node
from compliance_pipeline.schemas import (
    CheckBatchOutput, CheckItemOutput, EvidenceQuote, ItemResult, Status, SubCriterionResult,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(retrieval, "get_rules", lambda _: [])
    monkeypatch.setenv("COMPLIANCE_LLM_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("COMPLIANCE_CHECK_RECOVERY_BATCH_SIZE", "1")
    monkeypatch.setattr(llm, "TRANSIENT_BACKOFF_S", [])
    llm.set_backend(lambda *a: pytest.fail("A test must install its own fake model"))
    yield
    llm.reset_backend()


@pytest.mark.parametrize("text,quote,selected", [
    ("前文：本金可能全部亏损。后文", "本金 可能 全部 亏损", "本金可能全部亏损"),
    ("交易数量为５０股。", "交易数量为50股", "交易数量为５０股"),
    ("请买入 USD 产品。", "请 买入USD产品", "请买入 USD 产品"),
    ("Café is a label.", "CAFÉ", "Café"),
    ("Cafe\u0301 is a label.", "CAFÉ", "Cafe\u0301"),
    ("Straße is a label.", "STRASSE", "Straße"),
    ("I confirm, the instruction.", "i confirm the instruction", "I confirm, the instruction"),
    ("费用为５．５％。", "费用为5.5%", "费用为５．５％"),
])
def test_equivalent_quotes_map_to_exact_original_offsets(text, quote, selected):
    refs, errors = resolve_evidence([{"id": "seg_0001", "text": text}],
                                   [EvidenceQuote(segment_id="seg_0001", quote=quote)])
    assert not errors and len(refs) == 1 and refs[0].verified
    assert text[refs[0].start_char:refs[0].end_char] == selected


@pytest.mark.parametrize("text,quote", [
    ("本金不可能全部亏损", "本金可能全部亏损"),
    ("不要买入五百股", "买入五十股"),
    ("请确认卖出", "Please confirm the sale"),
    ("ＳＧＤ５０股", "SGD500股"),
    ("收费1.5%", "收费1,5%"),
    ("费用为50元", "费用为50%"),
    ("do not buy", "donotbuy"),
    ("ß", "s"),
    ("1\u20442", "1 2"),
    ("2\u00d75", "2 5"),
])
def test_matching_never_translates_changes_values_or_removes_negation(text, quote):
    assert _find_normalised(text, quote) is None


def answer(status="PASS", quote="本金 可能 全部 亏损"):
    return {"sub_criteria": [{"id": "met", "status": status,
                             "evidence": [{"segment_id": "seg_0001", "quote": quote}]
                             if status == "PASS" else [], "note": "Synthetic requirement review."}],
            "confidence": 0.95, "reasoning": "Synthetic response."}


def task(count=1):
    items = resolve_checklist([{"id": f"Q{i}", "question": f"Was required disclosure {i} explained?"}
                               for i in range(count)])
    return {"call_id": "synthetic", "checklist_items": [i.model_dump() for i in items],
            "segments": [{"id": "seg_0001", "speaker": "spk_0", "role": "Advisor",
                          "text": "本金可能全部亏损。", "start": 0, "end": 45, "word_count": 1},
                         {"id": "seg_0002", "speaker": "spk_1", "role": "Client",
                          "text": "我听明白了。", "start": 45, "end": 90, "word_count": 1}]}


def test_chinese_spacing_does_not_force_a_paid_format_repair():
    value, calls = task(), []
    value.update(item_id="Q0", checklist_item=value["checklist_items"][0])
    llm.set_backend(lambda *args: calls.append(args[0]) or answer())
    result = check_item_node(value)["item_results"][0]
    assert calls == ["check_item"]
    assert result.status == Status.PASS and result.execution_status == "success"
    assert result.evidence[0].verified


def test_missing_status_is_repaired_but_explicit_semantic_unclear_is_final():
    value, calls = task(), []
    value.update(item_id="Q0", checklist_item=value["checklist_items"][0])
    def backend(*args):
        calls.append(args[0])
        row = answer("CANNOT_DETERMINE")
        if len(calls) == 1:
            row["sub_criteria"][0].pop("status")
        return row
    llm.set_backend(backend)
    result = check_item_node(value)["item_results"][0]
    assert calls == ["check_item", "check_item"]
    assert result.status == Status.CANNOT_DETERMINE and result.execution_status == "success"


@pytest.mark.parametrize("malformation", ["missing_status", "invalid_status"])
def test_partial_provider_response_retains_fail_and_unclear_and_repairs_only_malformed_sibling(monkeypatch, malformation):
    value, calls = task(20), []
    payload = {"answers": [{"item_id": f"Q{i}", "result": answer()} for i in range(20)]}
    payload["answers"][0]["result"] = answer("FAIL")
    payload["answers"][1]["result"] = answer("CANNOT_DETERMINE")
    if malformation == "missing_status":
        payload["answers"][2]["result"]["sub_criteria"][0].pop("status")
    else:
        payload["answers"][2]["result"]["sub_criteria"][0]["status"] = "invalid"
    with pytest.raises(ValidationError) as caught:
        CheckBatchOutput.model_validate(payload)

    class Client:
        def __init__(self, stage):
            self.stage = stage
        def with_structured_output(self, schema, *, method, include_raw=False):
            def invoke(messages):
                calls.append((self.stage, messages))
                if self.stage == "check_batch":
                    assert include_raw
                    return {"parsed": None, "parsing_error": caught.value,
                            "raw": AIMessage(content="", tool_calls=[{
                                "name": "CheckBatchOutput", "args": payload, "id": "synthetic"}])}
                assert "disclosure 2" in messages[1][1]
                assert "disclosure 0" not in messages[1][1] and "disclosure 1?" not in messages[1][1]
                return schema.model_validate(answer())
            return SimpleNamespace(invoke=invoke)

    monkeypatch.setattr(llm, "_get_client", lambda model, stage: Client(stage))
    llm.reset_backend()  # the real parser is exercised against the fake client above
    result = check_batch_node(value)
    assert [c[0] for c in calls] == ["check_batch", "check_item"]
    rows = {r.item_id: r for r in result["item_results"]}
    assert len(rows) == 20
    assert rows["Q0"].status == Status.FAIL
    assert rows["Q1"].status == Status.CANNOT_DETERMINE
    assert rows["Q2"].status == Status.PASS
    assert all(r.execution_status == "success" for r in rows.values())


def test_partial_recovery_never_hides_duplicate_ids_or_invents_missing_status():
    good = {"item_id": "good", "result": answer()}
    dup = {"item_id": "duplicate", "result": answer()}
    bad = deepcopy(dup)
    bad["result"]["sub_criteria"][0].pop("status")
    result = recover_grouped_output("CheckBatchOutput", {"answers": [good, dup, bad]}, CheckBatchOutput)
    assert [row.item_id for row in result.answers] == ["good"]
    assert recover_grouped_output("wrong_tool", {"answers": [good]}, CheckBatchOutput) is None
    assert recover_grouped_output("CheckBatchOutput", {"answers": [bad]}, CheckBatchOutput) is None
    assert recover_grouped_output("CheckBatchOutput", {"answers": [good]}, CheckItemOutput) is None


def test_native_batch_partial_recovery_does_not_resubmit_or_double_count_usage(monkeypatch):
    from google.genai import types
    from compliance_pipeline.usage import track_usage
    payload = {"answers": [{"item_id": "good", "result": answer()},
                           {"item_id": "bad", "result": answer()}]}
    payload["answers"][1]["result"]["sub_criteria"][0]["status"] = "invalid"
    response = types.GenerateContentResponse(
        model_version="fake-gemini",
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=100, candidates_token_count=20, total_token_count=120),
        candidates=[types.Candidate(content=types.Content(role="model", parts=[
            types.Part(function_call=types.FunctionCall(name="CheckBatchOutput", args=payload))]))])
    submitted = []
    def invoke(model, request):
        submitted.append(request)
        return SimpleNamespace(response=response, error=None), "batches/synthetic"
    monkeypatch.setattr(llm, "_get_native_transport", lambda: SimpleNamespace(invoke=invoke))
    @track_usage
    def node(_):
        return {"result": llm._native_backend("check_batch", "system", "transcript", CheckBatchOutput, "fake-gemini")}
    result = node({})
    assert len(submitted) == 1
    assert [row.item_id for row in result["result"].answers] == ["good"]
    assert result["usage"]["fake-gemini"]["input_tokens"] == 100


@pytest.mark.parametrize("quote", ["invented words", ""])
def test_verification_does_not_downgrade_a_failure_using_invalid_counter_evidence(quote):
    value, calls = task(), []
    value["checklist"] = value["checklist_items"]
    value["item_results"] = [ItemResult(item_id="Q0", status=Status.FAIL, confidence=0.95,
        sub_criteria=[SubCriterionResult(id="met", status=Status.FAIL, note="Required act omitted.")])]
    def backend(*args):
        calls.append(args[0])
        return {"verdicts": [{"target_id": "met", "verdict": "not_supported" if len(calls) == 1 else "upheld",
                "counter_evidence": [{"segment_id": "seg_0001", "quote": quote}] if len(calls) == 1 else []}]}
    llm.set_backend(backend)
    result = verify_node(value)["verified_results"][0]
    assert calls == ["verify", "verify"]
    assert result.status == Status.FAIL and result.execution_status == "success" and result.verified


def test_valid_verifier_uncertainty_is_not_retried_or_relabelled():
    value, calls = task(), []
    value["checklist"] = value["checklist_items"]
    value["item_results"] = [ItemResult(item_id="Q0", status=Status.FAIL, confidence=0.95,
        sub_criteria=[SubCriterionResult(id="met", status=Status.FAIL, note="Required act omitted.")])]
    llm.set_backend(lambda *args: calls.append(args[0]) or {
        "verdicts": [{"target_id": "met", "verdict": "not_supported", "reason": "Context is insufficient."}]})
    result = verify_node(value)["verified_results"][0]
    assert calls == ["verify"] and result.status == Status.CANNOT_DETERMINE


@pytest.mark.parametrize("timing", [(0, 0), (None, None), (float("nan"), 10), (10, 5)])
def test_placeholder_or_invalid_timing_does_not_claim_a_short_recording(timing):
    value = task()
    for segment in value["segments"]:
        segment["start"], segment["end"] = timing
    flags = quality_check_node(value)["quality_flags"]
    assert "timing_unavailable" in flags and "short_call" not in flags and "empty_transcript" not in flags


def test_real_short_and_long_recordings_keep_their_existing_quality_check():
    value = task()
    assert "short_call" not in quality_check_node(value)["quality_flags"]
    value["segments"][0]["end"] = 10
    value["segments"][1].update(start=10, end=20)
    assert "short_call" in quality_check_node(value)["quality_flags"]
