import json
import uuid
import pytest
from compliance_pipeline import observability as obs


@pytest.fixture(autouse=True)
def synthetic_prices(tmp_path, monkeypatch):
    path = tmp_path / "synthetic-prices.json"
    path.write_text(json.dumps({"currency": "USD", "valid_until": "2099-12-31", "models": {
        "synthetic": {"standard": {"input": 1, "output": 2, "cached_input": .1}}}}))
    monkeypatch.setenv("COMPLIANCE_PRICE_BOOK_PATH", str(path))
    monkeypatch.setenv("COMPLIANCE_OBSERVABILITY_ENABLED", "1")


def state():
    return {**obs.begin_observation({"analysis_run_id": str(uuid.uuid4())}),
            "transcript": "PRIVATE_CLIENT_TRANSCRIPT", "final_results": [
                {"item_id": "arbitrary", "status": "FAIL", "reasoning": "PRIVATE_CLIENT_REASON", "execution_status": "success"}],
            "usage_by_transport": {"standard": {"synthetic": {
                "input_tokens": 100, "output_tokens": 50, "input_token_details": {"cache_read": 20},
                "output_token_details": {"reasoning": 30}}}}}


def test_local_telemetry_excludes_client_text_and_prices_reasoning_once():
    value = state()
    result = obs.finish_observation(value)["observability"]
    assert "PRIVATE_CLIENT" not in json.dumps(result)
    assert result["status"] == "completed"  # A breach is not a runtime error.
    assert result["prediction_failures"] == 0
    assert result["cost"]["amount"] == ((80 * 1) + (20 * .1) + (50 * 2)) / 1_000_000
    assert result["trace_id"] == uuid.UUID(value["analysis_run_id"]).hex
    assert result["duration_ms"] >= 0


def test_semantic_unclear_is_distinct_from_failed_prediction():
    value = state()
    value["final_results"] = [{"status": "CANNOT_DETERMINE", "execution_status": "success"},
                              {"status": "CANNOT_DETERMINE", "execution_status": "prediction_failure"}]
    result = obs.finish_observation(value)["observability"]
    assert result["status"] == "partial" and result["prediction_failures"] == 1


def test_disabled_metrics_do_not_change_node_updates_or_invoke_extra_work(monkeypatch):
    monkeypatch.setenv("COMPLIANCE_OBSERVABILITY_ENABLED", "0")
    calls = []
    def node(value):
        calls.append(value)
        return {"answer": "unchanged"}
    assert obs.track_stage(node)({}) == {"answer": "unchanged"}
    assert len(calls) == 1
    assert obs.begin_observation({}) == {} and obs.finish_observation(state()) == {}


def test_unpriced_or_absent_usage_is_unknown_not_zero():
    value = state(); value["usage_by_transport"] = {}
    assert obs.finish_observation(value)["observability"]["cost"]["amount"] is None
    value["usage_by_transport"] = {"standard": {"unknown-model": {"input_tokens": 50, "output_tokens": 20}}}
    assert obs.finish_observation(value)["observability"]["cost"]["amount"] is None


def test_stage_records_keep_only_safe_error_codes_and_usage():
    result = obs.stage_record("verify", 1000000000, 1100000000, calls=[
        {"attempts": 2, "status": "prediction_failure", "error": "Private provider details (code 429)"}])
    assert result["error_codes"] == ["429"] and result["attempts"] == 2
    assert result["duration_ms"] == 100
    assert "Private" not in json.dumps(result)
