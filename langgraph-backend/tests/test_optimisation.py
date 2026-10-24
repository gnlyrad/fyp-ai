"""Cost and decision-coverage regressions using fake models, never live inference."""
from collections import Counter

import pytest

from compliance_pipeline import llm, retrieval
from compliance_pipeline.config import resolve_checklist
from compliance_pipeline.graph import _route_to_checks
from compliance_pipeline.nodes.check_batch import check_batch_node, check_batch_size
from compliance_pipeline.nodes.check_item import check_item_node
from compliance_pipeline.schemas import Status


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(retrieval, "get_rules", lambda item: [])
    monkeypatch.setenv("COMPLIANCE_LLM_MAX_ATTEMPTS", "2")
    monkeypatch.setenv("COMPLIANCE_CHECK_BATCH_SIZE", "1")
    yield
    llm.reset_backend()


def task(count=3):
    return {"checklist_items": [item.model_dump() for item in resolve_checklist([
        {"id": f"Q{i}", "question": f"Was requirement {i} met?"} for i in range(count)])],
        "segments": [{"id": "seg_0001", "text": "I confirm.", "role": "Client"}],
        "call_intent": None}


def pass_answer():
    return {"sub_criteria": [{"id": "met", "status": "PASS",
                              "evidence": [{"segment_id": "seg_0001", "quote": "I confirm."}]}],
            "confidence": 0.95}


@pytest.mark.parametrize("count", [5, 20])
def test_grouping_preserves_every_answer_and_reduces_request_count(count):
    value, calls = task(count), []
    def backend(name, system, user, schema, model):
        calls.append(name)
        return schema.model_validate({"answers": [
            {"item_id": item["id"], "result": pass_answer()}
            for item in reversed(value["checklist_items"])]})
    llm.set_backend(backend)
    output = check_batch_node(value)
    assert calls == ["check_batch"]
    assert [item.item_id for item in output["item_results"]] == [f"Q{i}" for i in range(count)]
    assert all(item.status == Status.PASS and item.evidence[0].verified for item in output["item_results"])


def test_only_missing_or_duplicate_group_answers_are_retried():
    value, calls = task(), []
    def backend(name, system, user, schema, model):
        calls.append(name)
        if name == "check_batch":
            return schema.model_validate({"answers": [
                {"item_id": "Q0", "result": pass_answer()},
                {"item_id": "Q1", "result": pass_answer()},
                {"item_id": "Q1", "result": pass_answer()}]})
        assert "requirement 0" not in user
        return schema.model_validate(pass_answer())
    llm.set_backend(backend)
    output = check_batch_node(value)
    assert Counter(calls) == {"check_batch": 1, "check_item": 2}
    assert len(output["item_results"]) == 3
    assert all(item.status == Status.PASS for item in output["item_results"])


def test_unset_batch_flag_routes_each_question_individually(monkeypatch):
    monkeypatch.delenv("COMPLIANCE_CHECK_BATCH_SIZE", raising=False)
    value = task(20)
    routes = _route_to_checks({"checklist": value["checklist_items"],
                               "segments": value["segments"], "call_id": "one-call"})
    assert check_batch_size() == 1
    assert len(routes) == 20
    assert all(route.node == "check_item" and route.arg["call_id"] == "one-call" for route in routes)
    assert [route.arg["item_id"] for route in routes] == [item["id"] for item in value["checklist_items"]]


def test_graph_routes_twenty_questions_as_four_groups(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_CHECK_BATCH_SIZE", "5")
    value = task(20)
    routes = _route_to_checks({"checklist": value["checklist_items"], "segments": value["segments"]})
    assert len(routes) == 4
    assert all(route.node == "check_batch" and len(route.arg["checklist_items"]) == 5 for route in routes)


def test_large_batch_routes_all_items_including_a_short_final_group(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_CHECK_BATCH_SIZE", "20")
    value = task(23)
    routes = _route_to_checks({"checklist": value["checklist_items"], "segments": value["segments"]})
    assert check_batch_size() == 20
    assert [len(route.arg["checklist_items"]) for route in routes] == [20, 3]
    assert [item["id"] for route in routes for item in route.arg["checklist_items"]] == [
        item["id"] for item in value["checklist_items"]]
    monkeypatch.setenv("COMPLIANCE_CHECK_BATCH_SIZE", "21")
    with pytest.raises(ValueError):
        check_batch_size()


def test_shared_rules_preserve_item_scope_and_distinct_passages():
    from compliance_pipeline.prompts.check_item import build_batch_user
    from compliance_pipeline.schemas import ChecklistItem

    value = task(3)
    items = [ChecklistItem.model_validate(item) for item in value["checklist_items"]]
    common = {"rule_id": "R", "source": "Rulebook", "page": 7, "text": "A shared regulatory passage."}
    distinct = {**common, "text": "A different regulatory passage."}
    prompt = build_batch_user(items, {"Q0": [common, common], "Q1": [common, distinct], "Q2": []},
                              "I confirm.", {"category": "advisory"})
    assert prompt.count(common["text"]) == prompt.count(distinct["text"]) == 1
    assert prompt.count("I confirm.") == 1
    q0, rest = prompt.split("ITEM Q0\n", 1)[1].split("ITEM Q1\n")
    q1, q2 = rest.split("ITEM Q2\n")
    assert "EXTRACT_1 [rule R]" in q0 and "EXTRACT_2" not in q0
    assert "EXTRACT_1 [rule R]" in q1 and "EXTRACT_2 [rule R]" in q1
    assert "No regulatory extracts were available" in q2 and "EXTRACT_1" not in q2


@pytest.mark.parametrize("bad_quote", [
    {"segment_id": "seg_001", "quote": "I confirm."},
    {"segment_id": "seg_0001", "quote": "I never said this."},
])
def test_bad_evidence_is_repaired_before_it_can_reduce_coverage(bad_quote):
    value, calls = task(1), []
    value.update(item_id="Q0", checklist_item=value["checklist_items"][0])
    def backend(name, system, user, schema, model):
        calls.append(user)
        answer = pass_answer()
        if len(calls) == 1:
            answer["sub_criteria"][0]["evidence"] = [bad_quote]
        return answer
    llm.set_backend(backend)
    output = check_item_node(value)
    assert len(calls) == 2 and "Output validation failed" in calls[1]
    assert output.get("errors", []) == []
    assert output["item_results"][0].status == Status.PASS
    assert output["item_results"][0].evidence[0].verified


def test_grouped_bad_evidence_retries_only_the_affected_item():
    value, calls = task(2), []
    def backend(name, system, user, schema, model):
        calls.append(name)
        if name == "check_batch":
            bad = pass_answer()
            bad["sub_criteria"][0]["evidence"][0]["segment_id"] = "seg_001"
            return {"answers": [{"item_id": "Q0", "result": pass_answer()},
                                {"item_id": "Q1", "result": bad}]}
        assert "requirement 0" not in user
        return pass_answer()
    llm.set_backend(backend)
    output = check_batch_node(value)
    assert calls == ["check_batch", "check_item"]
    assert output["errors"] == []
    assert all(item.status == Status.PASS for item in output["item_results"])


def test_missing_subcriterion_is_repaired_but_semantic_unclear_is_not_retried():
    value = task(1)
    value.update(item_id="Q0", checklist_item=value["checklist_items"][0])
    calls = []
    def backend(name, system, user, schema, model):
        calls.append(user)
        return schema.model_validate({} if len(calls) == 1 else {
            "sub_criteria": [{"id": "met", "status": "CANNOT_DETERMINE"}]})
    llm.set_backend(backend)
    result = check_item_node(value)["item_results"][0]
    assert len(calls) == 2 and "Output validation failed" in calls[-1]
    assert result.status == Status.CANNOT_DETERMINE and result.execution_status == "success"
    calls.clear()
    llm.set_backend(lambda *args: calls.append(1) or {
        "sub_criteria": [{"id": "met", "status": "CANNOT_DETERMINE"}]})
    check_item_node(value)
    assert calls == [1]


def test_prohibition_metadata_allows_absence_without_weakening_positive_evidence():
    item = resolve_checklist([{"id": "P", "question": "No guarantee was made.",
                               "requirement_type": "PROHIBITION"}])[0]
    assert item.requirement_type == "PROHIBITION"
    assert item.sub_criteria[0].absence
    llm.set_backend(lambda *args: {"sub_criteria": [{"id": "met", "status": "PASS"}], "confidence": 0.95})
    value = {**task(1), "item_id": "P", "checklist_item": item.model_dump()}
    assert check_item_node(value)["item_results"][0].status == Status.PASS
    item.sub_criteria[0].absence = False
    value["checklist_item"] = item.model_dump()
    assert check_item_node(value)["item_results"][0].status == Status.CANNOT_DETERMINE


@pytest.mark.parametrize("metadata", [{}, {"requirement_type": None},
                                     {"requirement_type": "OBLIGATION"},
                                     {"requirement_type": "PROHIBITION"}])
def test_optional_production_requirement_type_is_preserved_without_inference(metadata):
    request = {"id": "custom", "question": "Was this requirement met?", **metadata}
    original = dict(request)
    item = resolve_checklist([request])[0]
    assert item.requirement_type == metadata.get("requirement_type")
    assert item.sub_criteria[0].absence is (metadata.get("requirement_type") == "PROHIBITION")
    assert request == original


def test_stage_model_settings_do_not_reuse_incompatible_clients(monkeypatch):
    clients = []
    monkeypatch.setattr(llm, "_clients", {})
    monkeypatch.setattr(llm, "create_chat_model", lambda **kw: clients.append(kw) or object())
    monkeypatch.setenv("COMPLIANCE_CHECK_BATCH_MAX_OUTPUT_TOKENS", "8192")
    monkeypatch.setenv("COMPLIANCE_CHECK_BATCH_THINKING_LEVEL", "low")
    one = llm._get_client("test-model", "check_item")
    two = llm._get_client("test-model", "check_batch")
    assert one is not two and llm._get_client("test-model", "check_batch") is two
    assert clients[-1]["thinking_level"] == "low" and clients[-1]["max_output_tokens"] == 8192


def test_missing_verifier_target_gets_one_repair_attempt():
    from compliance_pipeline.nodes.verify import verify_node
    from compliance_pipeline.schemas import ItemResult, SubCriterionResult
    value, calls = task(1), []
    value["checklist"] = value["checklist_items"]
    value["item_results"] = [ItemResult(item_id="Q0", status=Status.FAIL,
        sub_criteria=[SubCriterionResult(id="met", status=Status.FAIL, note="Omitted.")])]
    def backend(name, system, user, schema, model):
        calls.append(name)
        return {} if len(calls) == 1 else {
            "verdicts": [{"target_id": "met", "verdict": "upheld"}]}
    llm.set_backend(backend)
    result = verify_node(value)["verified_results"][0]
    assert calls == ["verify", "verify"] and result.verified
    assert result.status == Status.FAIL and result.execution_status == "success"


@pytest.mark.parametrize("bad_quote", [
    {"segment_id": "seg_999", "quote": "I confirm."},
    {"segment_id": "seg_0001", "quote": "I never said this."},
])
def test_action_citations_are_repaired_without_repeating_checks(bad_quote):
    from compliance_pipeline.nodes.decide import decide_node
    value, calls = task(1), []
    value["checklist"] = value["checklist_items"]
    def backend(name, system, user, schema, model):
        calls.append((name, user))
        return {"actions": [{"action_type": "place_order", "description": "Process the instruction.",
                             "evidence": [bad_quote if len(calls) == 1 else {
                                 "segment_id": "seg_0001", "quote": "I confirm."}]}]}
    llm.set_backend(backend)
    result = decide_node(value)
    assert [name for name, _ in calls] == ["decide_actions", "decide_actions"]
    assert "Output validation failed" in calls[-1][1]
    assert result.get("errors", []) == []
    assert result["actions"][0].evidence[0].verified
    assert "item_results" not in result and "verified_results" not in result


@pytest.mark.parametrize("alias", ["seg_1", "seg_01", "seg_001"])
def test_action_zero_padding_is_repaired_locally_only_with_exact_quote(alias):
    from compliance_pipeline.nodes.decide import decide_node
    value, calls = task(1), []
    value["checklist"] = value["checklist_items"]
    def backend(name, system, user, schema, model):
        calls.append(name)
        return {"actions": [{"action_type": "place_order", "description": "Process the instruction.",
                             "evidence": [{"segment_id": alias, "quote": "I confirm."}]}]}
    llm.set_backend(backend)
    result = decide_node(value)
    assert calls == ["decide_actions"]
    assert not result.get("errors")
    assert result["actions"][0].evidence[0].segment_id == "seg_0001"
    assert result["actions"][0].evidence[0].verified


@pytest.mark.parametrize("segments,quote", [
    ([{"id": "seg_0001", "text": "I confirm."}, {"id": "seg_01", "text": "Different turn."}], "I confirm."),
    ([{"id": "seg_0001", "text": "I confirm."}], "Unspoken words."),
])
def test_action_alias_rejects_ambiguous_ids_and_unmatched_quotes(segments, quote):
    from compliance_pipeline.nodes.decide import _actions_output_model, _validate_actions
    from compliance_pipeline.reliability import InvalidStructuredOutput
    output = _actions_output_model().model_validate({"actions": [{"action_type": "place_order",
        "description": "Process the instruction.", "evidence": [{"segment_id": "seg_001", "quote": quote}]}]})
    with pytest.raises(InvalidStructuredOutput):
        _validate_actions(output, segments)


def test_unrepairable_action_citation_remains_an_explicit_failure():
    from compliance_pipeline.nodes.decide import decide_node
    value, calls = task(1), []
    value["checklist"] = value["checklist_items"]
    def backend(name, system, user, schema, model):
        calls.append(name)
        return {"actions": [{"action_type": "place_order", "description": "Process the instruction.",
                             "evidence": [{"segment_id": "invented", "quote": "I confirm."}]}]}
    llm.set_backend(backend)
    result = decide_node(value)
    assert calls == ["decide_actions", "decide_actions", "decide_actions_repair", "decide_actions_repair"]
    assert not result["actions"]
    assert result["errors"] == ["decide[actions]: InvalidStructuredOutput"]


def test_higher_effort_action_fallback_runs_only_after_invalid_output(monkeypatch):
    from compliance_pipeline.nodes.decide import decide_node
    value, calls = task(1), []
    value["checklist"] = value["checklist_items"]
    def backend(name, system, user, schema, model):
        calls.append(name)
        return {"actions": [{"action_type": "place_order", "description": "Process the instruction.",
                             "evidence": [{"segment_id": "seg_0001" if name.endswith("_repair") else "invented",
                                           "quote": "I confirm."}]}]}
    llm.set_backend(backend)
    result = decide_node(value)
    assert calls == ["decide_actions", "decide_actions", "decide_actions_repair"]
    assert not result.get("errors") and result["actions"][0].evidence[0].verified
    clients = []
    monkeypatch.setattr(llm, "_clients", {})
    monkeypatch.setattr(llm, "create_chat_model", lambda **kw: clients.append(kw) or object())
    monkeypatch.delenv("COMPLIANCE_THINKING_LEVEL", raising=False)
    monkeypatch.delenv("COMPLIANCE_DECIDE_ACTIONS_REPAIR_THINKING_LEVEL", raising=False)
    llm._get_client("test-model", "decide_actions_repair")
    assert clients[-1]["thinking_level"] == "medium"


def absence_answer(**review_updates):
    review = {"prohibited_conduct": "promising guaranteed investment returns",
              "entire_transcript_reviewed": True, "context_sufficient": True}
    review.update(review_updates)
    return {"sub_criteria": [{"id": "met", "status": "PASS", "absence_review": review,
                             "note": "Reviewed the full call; no guaranteed returns were promised."}],
            "confidence": 0.95}


@pytest.mark.parametrize("grouped", [False, True])
def test_id_question_only_absence_has_auditable_basis_without_metadata_inference(grouped):
    item = resolve_checklist([{"id": "custom_negative", "question":
        "The employee must not promise guaranteed investment returns."}])[0]
    before = item.model_dump()
    value = {**task(1), "item_id": item.id, "checklist_item": before,
             "checklist_items": [before]}
    calls = []
    def backend(name, system, user, schema, model):
        calls.append(name)
        answer = absence_answer()
        return {"answers": [{"item_id": item.id, "result": answer}]} if grouped else answer
    llm.set_backend(backend)
    result = (check_batch_node(value) if grouped else check_item_node(value))["item_results"][0]
    assert calls == ["check_batch" if grouped else "check_item"]
    assert result.status == Status.PASS and result.execution_status == "success"
    assert not result.evidence
    assert result.sub_criteria[0].absence_review.prohibited_conduct == "promising guaranteed investment returns"
    assert item.model_dump() == before and item.requirement_type is None
    assert not item.sub_criteria[0].absence


@pytest.mark.parametrize("update,flags,note", [
    ({"entire_transcript_reviewed": False}, [], None),
    ({"context_sufficient": False}, [], None),
    ({"prohibited_conduct": "  "}, [], None),
    ({}, ["empty_transcript"], None),
    ({}, ["low_confidence"], None),
    ({}, ["single_speaker"], None),
    ({}, [], ""),
])
def test_incomplete_absence_review_does_not_force_a_pass_or_trigger_semantic_retry(update, flags, note):
    value = task(1)
    value.update(item_id="Q0", checklist_item=value["checklist_items"][0], quality_flags=flags)
    answer = absence_answer(**update)
    if note is not None:
        answer["sub_criteria"][0]["note"] = note
    calls = []
    llm.set_backend(lambda *args: calls.append(1) or answer)
    result = check_item_node(value)["item_results"][0]
    assert calls == [1]
    assert result.status == Status.CANNOT_DETERMINE and result.execution_status == "success"
    assert result.sub_criteria[0].absence_review is None


def test_absence_of_required_consent_cannot_pass_an_explicit_obligation():
    item = resolve_checklist([{"id": "consent", "question": "Obtain the client's consent.",
                               "requirement_type": "OBLIGATION"}])[0]
    value = {**task(1), "item_id": item.id, "checklist_item": item.model_dump()}
    answer = absence_answer(prohibited_conduct="missing consent")
    llm.set_backend(lambda *args: answer)
    assert check_item_node(value)["item_results"][0].status == Status.CANNOT_DETERMINE


@pytest.mark.parametrize("verdict,expected", [("upheld", Status.PASS),
                                             ("not_supported", Status.CANNOT_DETERMINE)])
def test_absence_finding_survives_normal_verifier_contract(verdict, expected):
    from compliance_pipeline.nodes.verify import verify_node
    value = task(1)
    value.update(item_id="Q0", checklist_item=value["checklist_items"][0])
    llm.set_backend(lambda *args: absence_answer())
    item = check_item_node(value)["item_results"][0]
    item.confidence = 0.6
    calls = []
    def backend(name, system, user, schema, model):
        calls.append(name)
        assert "absence_review:" in user and "promising guaranteed investment returns" in user
        return {"verdicts": [{"target_id": "met", "verdict": verdict,
                              "reason": "Full transcript reviewed."}]}
    llm.set_backend(backend)
    result = verify_node({**value, "checklist": value["checklist_items"],
                          "item_results": [item]})["verified_results"][0]
    assert calls == ["verify"] and result.status == expected
    assert bool(result.sub_criteria[0].absence_review) is (verdict == "upheld")
    assert item.status == Status.PASS and item.sub_criteria[0].absence_review is not None


def test_grouped_recovery_retains_successes_and_never_retries_semantic_answers(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_CHECK_RECOVERY_BATCH_SIZE", "20")
    value, calls = task(20), []
    unclear = {"sub_criteria": [{"id": "met", "status": "CANNOT_DETERMINE"}]}
    failed = {"sub_criteria": [{"id": "met", "status": "FAIL", "note": "Required act omitted."}]}
    def backend(name, system, user, schema, model):
        calls.append((name, user))
        if len(calls) == 1:
            return {"answers": [{"item_id": "Q0", "result": pass_answer()},
                                {"item_id": "Q1", "result": unclear},
                                {"item_id": "Q2", "result": failed}]}
        assert name == "check_batch"
        assert all(f"ITEM Q{i}\n" not in user for i in range(3))
        return {"answers": [{"item_id": f"Q{i}", "result": pass_answer()} for i in range(3, 20)]}
    llm.set_backend(backend)
    output = check_batch_node(value)
    assert len(calls) == 2
    assert [row.item_id for row in output["item_results"]] == [f"Q{i}" for i in range(20)]
    assert [row.status for row in output["item_results"][:3]] == [
        Status.PASS, Status.CANNOT_DETERMINE, Status.FAIL]
    assert all(row.execution_status == "success" for row in output["item_results"])


@pytest.mark.parametrize("size,expected", [(2, [2, 2]), (3, [3, 2]), (20, [5])])
def test_recovery_groups_are_bounded_and_retain_valid_siblings(monkeypatch, size, expected):
    monkeypatch.setenv("COMPLIANCE_CHECK_RECOVERY_BATCH_SIZE", str(size))
    value, groups = task(6), []
    def backend(name, system, user, schema, model):
        if not groups:
            groups.append([0])
            return {"answers": [{"item_id": "Q0", "result": pass_answer()}]}
        if name == "check_item":
            assert "requirement 5" in user
            return pass_answer()
        ids = [i for i in range(1, 6) if f"ITEM Q{i}\n" in user]
        groups.append(ids)
        return {"answers": [{"item_id": f"Q{i}", "result": pass_answer()} for i in reversed(ids)]}
    llm.set_backend(backend)
    output = check_batch_node(value)
    assert [len(group) for group in groups[1:]] == expected
    assert all(row.status == Status.PASS for row in output["item_results"])


def test_grouped_recovery_falls_back_only_for_still_unusable_rows(monkeypatch, caplog):
    monkeypatch.setenv("COMPLIANCE_CHECK_RECOVERY_BATCH_SIZE", "20")
    value, calls = task(4), []
    secret_quote = "words not present in the transcript"
    def backend(name, system, user, schema, model):
        calls.append(name)
        if len(calls) == 1:
            bad = pass_answer()
            bad["sub_criteria"][0]["evidence"][0]["quote"] = secret_quote
            return {"answers": [{"item_id": "Q0", "result": pass_answer()},
                                {"item_id": "Q1", "result": bad},
                                {"item_id": "Q2", "result": pass_answer()},
                                {"item_id": "Q2", "result": pass_answer()}]}
        if name == "check_batch":
            assert "ITEM Q0\n" not in user
            return {"answers": [{"item_id": "Q1", "result": pass_answer()},
                                {"item_id": "Q3", "result": pass_answer()}]}
        assert "requirement 2" in user
        return pass_answer()
    llm.set_backend(backend)
    output = check_batch_node(value)
    assert calls == ["check_batch", "check_batch", "check_item"]
    assert all(row.status == Status.PASS for row in output["item_results"])
    assert output["errors"] == []
    assert "invalid_structure_or_evidence" in caplog.text
    assert "missing_or_duplicate" in caplog.text
    assert secret_quote not in caplog.text


@pytest.mark.parametrize("native", [False, True])
def test_nonretryable_recovery_never_resubmits_and_preserves_good_sibling(monkeypatch, native):
    monkeypatch.setenv("COMPLIANCE_CHECK_RECOVERY_BATCH_SIZE", "20")
    if native:
        monkeypatch.setenv("COMPLIANCE_CHECK_TRANSPORT", "batch")
    value, calls = task(3), []
    class PendingOrPermanent(RuntimeError):
        no_retry = True
    def backend(name, system, user, schema, model):
        calls.append(name)
        if len(calls) == 1:
            return {"answers": [{"item_id": "Q0", "result": pass_answer()}]}
        raise PendingOrPermanent("Safe test error")
    llm.set_backend(backend)
    output = check_batch_node(value)
    assert calls == ["check_batch", "check_batch"]
    assert output["item_results"][0].status == Status.PASS
    assert all(row.execution_status == "prediction_failure" for row in output["item_results"][1:])
    assert len(output["errors"]) == 2


def test_recovery_pass_cannot_recurse_and_item_failure_remains_visible(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_CHECK_RECOVERY_BATCH_SIZE", "20")
    value, calls = task(2), []
    def backend(name, system, user, schema, model):
        calls.append(name)
        return {"answers": []} if name == "check_batch" else {}
    llm.set_backend(backend)
    output = check_batch_node(value)
    assert calls == ["check_batch", "check_batch", "check_item", "check_item", "check_item", "check_item"]
    assert all(row.execution_status == "prediction_failure" for row in output["item_results"])
    assert len(output["errors"]) == 2


@pytest.mark.parametrize("value", ["0", "21", "true", "-1"])
def test_invalid_recovery_setting_fails_before_model_call(monkeypatch, value):
    from compliance_pipeline.nodes.check_batch import check_recovery_batch_size
    monkeypatch.setenv("COMPLIANCE_CHECK_RECOVERY_BATCH_SIZE", value)
    llm.set_backend(lambda *args: pytest.fail("Invalid configuration must not call a model"))
    with pytest.raises(ValueError):
        check_recovery_batch_size()
