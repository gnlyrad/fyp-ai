"""Where per-model prices come from, and keeping them when they change."""
from datetime import date
from pathlib import Path

import pytest

from fyp_evaluation_engine.runner import run_evaluation
from fyp_evaluation_engine.usage import (
    catalog_price,
    load_pricing,
    refresh_entry,
    resolve_pricing,
    summarize_run_usage,
)

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2026, 10, 4)

# The record shape of the real catalog: the Developer-API entry, resellers that charge more, and Vertex.
CATALOG = {
    "sample_spec": {"input_cost_per_token": 0},
    "gemini/gemini-3.5-flash": {
        "litellm_provider": "gemini", "input_cost_per_token": 1.5e-06, "output_cost_per_token": 9e-06,
        "cache_read_input_token_cost": 1.5e-07, "source": "https://ai.google.dev/gemini-api/docs/pricing",
        "input_cost_per_token_batches": 7.5e-07, "output_cost_per_token_batches": 4.5e-06,
    },
    "databricks/databricks-gemini-3-5-flash": {"input_cost_per_token": 1.87502e-06, "output_cost_per_token": 1.124998e-05},
    "deepinfra/google/gemini-3.5-flash": {"input_cost_per_token": 9e-06, "output_cost_per_token": 9e-06},
    "gemini-3.5-flash": {"litellm_provider": "vertex_ai-language-models", "input_cost_per_token": 2e-06, "output_cost_per_token": 1e-05},
    "gpt-test": {"input_cost_per_token": 1e-06, "output_cost_per_token": 2e-06},
}


def _usage(input_tokens, output_tokens, cached=0):
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens,
            "input_token_details": {"cache_read": cached}, "output_token_details": {}}


def test_shipped_price_book_holds_no_unsourced_prices():
    for model, price in load_pricing(EVALUATION_ROOT / "model_pricing.json")["models"].items():
        assert price.get("source") and price.get("as_of"), f"{model} has a price with no source or date"


def test_catalog_prefers_the_gemini_developer_api_entry_and_converts_to_per_million():
    price = catalog_price(CATALOG, "gemini-3.5-flash", TODAY)
    assert (price["input"], price["output"], price["cached_input"]) == (1.5, 9.0, 0.15)  # not Vertex, not a reseller
    assert price["catalog_key"] == "gemini/gemini-3.5-flash"
    assert price["source"] == "https://ai.google.dev/gemini-api/docs/pricing" and price["as_of"] == "2026-10-04"


def test_catalog_handles_other_models_versioned_names_and_unknowns():
    assert catalog_price(CATALOG, "gpt-test", TODAY)["output"] == 2.0  # another provider needs no code change
    assert catalog_price(CATALOG, "gemini-3.5-flash-001", TODAY)["input"] == 1.5  # versioned name -> base model
    assert catalog_price(CATALOG, "no-such-model", TODAY) is None
    assert catalog_price(CATALOG, "sample_spec", TODAY) is None


def test_resolve_pricing_only_fills_models_without_an_override_and_never_mutates():
    override = {"models": {"gemini-3.5-flash": {"input": 1.0, "output": 2.0, "source": "contract", "as_of": "2026-01-01"}}}
    kept = resolve_pricing(override, {"gemini-3.5-flash"}, catalog=lambda: pytest.fail("an override must not hit the catalog"))
    assert kept["models"]["gemini-3.5-flash"]["source"] == "contract"

    empty = {"models": {}}
    filled = resolve_pricing(empty, {"gemini-3.5-flash", "gpt-test", "mystery"}, CATALOG)
    assert set(filled["models"]) == {"gemini-3.5-flash", "gpt-test"} and empty == {"models": {}}


def test_resolve_pricing_downloads_at_most_once_and_survives_being_offline():
    calls = []
    resolve_pricing({"models": {}}, {"gemini-3.5-flash", "gpt-test"}, lambda: calls.append(1) or CATALOG)
    assert calls == [1]
    offline = resolve_pricing({"models": {}}, {"gemini-3.5-flash"}, lambda: None)
    assert offline["models"] == {}  # unpriced, never zero


def test_a_null_override_falls_through_to_the_catalog():
    placeholder = {"models": {"gemini-3.5-flash": {"input": None, "output": None}}}
    assert resolve_pricing(placeholder, {"gemini-3.5-flash"}, CATALOG)["models"]["gemini-3.5-flash"]["input"] == 1.5


def test_run_cost_uses_the_catalog_price_and_records_where_it_came_from():
    pricing = resolve_pricing({"models": {}}, {"gemini-3.5-flash"}, CATALOG)
    summary = summarize_run_usage({"CALL_001": {"gemini-3.5-flash": _usage(1_000_000, 100_000, cached=200_000)}}, pricing)
    # 800k fresh input x $1.5 + 200k cached x $0.15 + 100k output x $9
    assert summary["estimated_cost"]["amount"] == pytest.approx(1.2 + 0.03 + 0.9)
    used = summary["estimated_cost"]["prices_used"]["gemini-3.5-flash"]
    assert used["source"].startswith("https://ai.google.dev") and used["as_of"] and "history" not in used


def test_a_price_drop_is_kept_in_history_and_earlier_runs_keep_their_own_price():
    first, change = refresh_entry(None, catalog_price(CATALOG, "gemini-3.5-flash", TODAY))
    assert change == "new" and first["history"] == []

    same, change = refresh_entry(first, catalog_price(CATALOG, "gemini-3.5-flash", date(2026, 11, 1)))
    assert change == "unchanged" and same["as_of"] == "2026-11-01"

    cheaper = {**CATALOG, "gemini/gemini-3.5-flash": {**CATALOG["gemini/gemini-3.5-flash"], "input_cost_per_token": 1e-06}}
    entry, change = refresh_entry(first, catalog_price(cheaper, "gemini-3.5-flash", date(2026, 12, 1)))
    assert change == "changed" and entry["input"] == 1.0
    assert entry["history"] == [{"input": 1.5, "output": 9.0, "cached_input": 0.15,
                                 "source": "https://ai.google.dev/gemini-api/docs/pricing", "as_of": "2026-10-04"}]

    # A run made before the drop recorded its own prices, so its cost does not change.
    old_run = summarize_run_usage({"CALL_001": {"gemini-3.5-flash": _usage(1_000_000, 0)}}, {"models": {"gemini-3.5-flash": first}})
    assert old_run["estimated_cost"]["amount"] == pytest.approx(1.5)


def test_runner_prices_a_run_from_the_catalog_when_no_override_exists(tmp_path, monkeypatch):
    monkeypatch.setattr("fyp_evaluation_engine.usage.fetch_catalog", lambda *args, **kwargs: CATALOG)

    def invoke(value):
        chunks = [{"chunk_id": f"c{i}", "source_segment_id": s["id"]} for i, s in enumerate(value["results"]["audio_segments"])]
        findings = [{"checklist_id": item["id"], "verdict": "Pass", "spans": []} for item in value["checklist"]]
        return {"chunks": chunks, "report": {"findings": findings, "summary": {}, "warnings": []},
                "usage": {"gemini-3.5-flash": _usage(1_000_000, 0)}}

    result = run_evaluation(EVALUATION_ROOT / "datasets" / "pilot", tmp_path, run_id="catalog-priced",
                            call_ids=["CALL_001"], graph_invoke=invoke)
    cost = result["metadata"]["observability"]["estimated_cost"]
    assert cost["amount"] == pytest.approx(1.5) and cost["unpriced_models"] == []
    assert cost["prices_used"]["gemini-3.5-flash"]["catalog"] == "LiteLLM price catalog"
