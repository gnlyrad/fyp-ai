"""Token usage and estimated cost of an evaluation run.

Usage is captured by LangChain's standard callback around the graph call, so the
pipeline itself is untouched. Chat-model tokens only: embedding calls are not counted.
Cost is tokens x per-model prices, and the prices used (with their source and date) are
copied into the run metadata, so history stays correct when a model later gets cheaper.

Where a price comes from, in order:
1. model_pricing.json, an optional hand-set override for a model;
2. otherwise a public price catalog (LiteLLM's community-maintained list, which cites each
   provider's own pricing page), fetched at run time only when a model has no override.
If neither has the model, tokens are still recorded and cost is left unavailable (never zero).
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

NOT_RECORDED = "not_exposed_by_current_graph_state"
PRICING_FILE = "model_pricing.json"
CATALOG_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
CATALOG_NAME = "LiteLLM price catalog"
# Catalog keys to try for a model id. The pipeline calls the Gemini Developer API, so that
# provider's entry comes first; resellers (databricks/, deepinfra/ ...) charge differently.
CATALOG_PREFIXES = ("gemini/", "", "vertex_ai/")
_TOKEN_FIELDS = ("input_tokens", "output_tokens", "total_tokens")


def invoke_with_usage(graph: Any, graph_input: dict[str, Any]) -> dict[str, Any]:
    """Run the graph and add `usage_metadata` ({model: tokens}) to its output.

    LangChain's callback totals tokens across every nested model call, including parallel
    branches and retried attempts. If the graph raises, the tokens already spent are attached
    to the exception as `usage_metadata` so a failed call is still costed.
    """
    from langchain_core.callbacks import UsageMetadataCallbackHandler  # noqa: PLC0415

    handler = UsageMetadataCallbackHandler()
    try:
        output = graph.invoke(graph_input, config={"callbacks": [handler]})
    except Exception as error:
        error.usage_metadata = dict(handler.usage_metadata)
        raise
    return {**output, "usage_metadata": dict(handler.usage_metadata)}


def model_key(name: str | None) -> str:
    """Normalise a reported model name ('models/gemini-x' -> 'gemini-x')."""
    return (name or "unknown").strip().removeprefix("models/")


def load_pricing(path: str | Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {"currency": "USD", "models": {}}


def _details(usage: dict[str, Any], group: str, name: str) -> int:
    value = (usage.get(group) or {}).get(name)
    return int(value) if isinstance(value, (int, float)) else 0


def normalize_usage(usage_by_model: dict[str, Any] | None) -> dict[str, dict[str, int]]:
    """{model: UsageMetadata} -> {model: flat integer counts}. Reasoning is already inside output_tokens."""
    result: dict[str, dict[str, int]] = {}
    for model, usage in (usage_by_model or {}).items():
        if not isinstance(usage, dict):
            continue
        entry = result.setdefault(model_key(model), dict.fromkeys((*_TOKEN_FIELDS, "reasoning_tokens", "cache_read_tokens"), 0))
        for field in _TOKEN_FIELDS:
            entry[field] += int(usage.get(field) or 0)
        entry["reasoning_tokens"] += _details(usage, "output_token_details", "reasoning")
        entry["cache_read_tokens"] += _details(usage, "input_token_details", "cache_read")
    return result


def _add(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + value


def _total(by_model: dict[str, dict[str, int]]) -> dict[str, int]:
    total = dict.fromkeys((*_TOKEN_FIELDS, "reasoning_tokens", "cache_read_tokens"), 0)
    for entry in by_model.values():
        _add(total, entry)
    return total


def usable(price: dict[str, Any] | None) -> bool:
    return bool(price) and price.get("input") is not None and price.get("output") is not None


def catalog_url() -> str | None:
    """EVALUATION_PRICE_CATALOG_URL overrides the source; 'off' disables the lookup."""
    url = os.getenv("EVALUATION_PRICE_CATALOG_URL", CATALOG_URL).strip()
    return None if url.lower() in {"", "off", "none"} else url


def fetch_catalog(timeout: float = 8.0) -> dict[str, Any] | None:
    """Download the price catalog, or None if disabled or unreachable (cost then stays unavailable)."""
    url = catalog_url()
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed https source
            return json.loads(response.read().decode("utf-8"))
    except Exception:
        return None


def catalog_price(catalog: dict[str, Any], model: str, today: date | None = None) -> dict[str, Any] | None:
    """Standard (non-batch, non-priority) per-million-token prices for a model, or None.

    Matches the exact id first, then the longest catalog id that prefixes a versioned name.
    """
    def entry(key: str) -> dict[str, Any] | None:
        value = catalog.get(key)
        return value if isinstance(value, dict) and isinstance(value.get("input_cost_per_token"), (int, float)) \
            and isinstance(value.get("output_cost_per_token"), (int, float)) else None

    found = next(((f"{prefix}{model}", entry(f"{prefix}{model}")) for prefix in CATALOG_PREFIXES if entry(f"{prefix}{model}")), None)
    if found is None:
        for prefix in CATALOG_PREFIXES:
            candidates = [key[len(prefix):] for key in catalog if key.startswith(prefix) and entry(key)
                          and (prefix or "/" not in key) and model.startswith(key[len(prefix):])]
            if candidates:
                best = f"{prefix}{max(candidates, key=len)}"
                found = (best, entry(best))
                break
    if found is None:
        return None
    key, record = found
    cached = record.get("cache_read_input_token_cost")

    def per_million(value: float) -> float:
        return round(value * 1_000_000, 6)

    return {
        "input": per_million(record["input_cost_per_token"]),
        "output": per_million(record["output_cost_per_token"]),
        "cached_input": per_million(cached) if isinstance(cached, (int, float)) else None,
        "source": record.get("source") or CATALOG_NAME,
        "catalog": CATALOG_NAME,
        "catalog_key": key,
        "as_of": (today or date.today()).isoformat(),
    }


def resolve_pricing(pricing: dict[str, Any], models: set[str] | list[str], catalog: Any = None) -> dict[str, Any]:
    """`pricing` plus a catalog price for every model that has no usable override.

    `catalog` may be a loaded catalog dict or a zero-argument loader (default: download once,
    and only if some model actually needs it). The input is not modified.
    """
    resolved = {**pricing, "models": dict(pricing.get("models") or {})}
    loaded = catalog if isinstance(catalog, dict) else None
    fetched = isinstance(catalog, dict)
    for model in sorted(models):
        if usable(price_for(model, resolved)):
            continue
        if not fetched:
            loaded = (catalog or fetch_catalog)()
            fetched = True
        found = catalog_price(loaded, model) if loaded else None
        if found:
            resolved["models"][model] = found
    return resolved


def refresh_entry(existing: dict[str, Any] | None, found: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """New price-book entry for a model, keeping the previous price in `history` when it changed.

    Returns (entry, change) where change is 'new', 'unchanged' or 'changed'.
    """
    rates = ("input", "output", "cached_input")
    if not usable(existing):
        return {**found, "history": list((existing or {}).get("history") or [])}, "new"
    if all(existing.get(key) == found.get(key) for key in rates):
        return {**existing, "as_of": found["as_of"]}, "unchanged"
    previous = {key: existing.get(key) for key in (*rates, "source", "as_of")}
    return {**found, "history": [*(existing.get("history") or []), previous]}, "changed"


def price_for(model: str, pricing: dict[str, Any]) -> dict[str, Any] | None:
    """Exact match first, then the longest priced id that prefixes a versioned name
    (the API may report 'gemini-3.5-flash-001' for the configured 'gemini-3.5-flash')."""
    prices = pricing.get("models") or {}
    if usable(prices.get(model)):
        return prices[model]
    candidates = [key for key in prices if model.startswith(key) and usable(prices[key])]
    return prices[max(candidates, key=len)] if candidates else None


def model_cost(tokens: dict[str, int], price: dict[str, Any] | None) -> float | None:
    """Cost of one model's tokens. Cached input is billed at its own rate when one is set."""
    if not usable(price):
        return None
    cached = tokens.get("cache_read_tokens", 0)
    cached_rate = price.get("cached_input")
    cached_rate = price["input"] if cached_rate is None else cached_rate
    fresh_input = max(tokens.get("input_tokens", 0) - cached, 0)
    return (fresh_input * price["input"] + cached * cached_rate + tokens.get("output_tokens", 0) * price["output"]) / 1_000_000


def summarize_run_usage(usage_by_call: dict[str, dict[str, Any]], pricing: dict[str, Any]) -> dict[str, Any]:
    """Run-level `observability` block from each call's {model: UsageMetadata}.

    Returns placeholders (not zeros) when nothing was recorded, so a missing measurement is
    never mistaken for a free run.
    """
    per_call = {call: normalize_usage(usage) for call, usage in usage_by_call.items()}
    per_call = {call: models for call, models in per_call.items() if models}
    if not per_call:
        return {"token_usage": NOT_RECORDED, "estimated_cost": NOT_RECORDED, "trace_ids": NOT_RECORDED}

    by_model: dict[str, dict[str, int]] = {}
    for models in per_call.values():
        for model, tokens in models.items():
            _add(by_model.setdefault(model, {}), tokens)
    prices = {model: price_for(model, pricing) for model in by_model}
    costs = {model: model_cost(by_model[model], prices[model]) for model in by_model}
    unpriced = sorted(model for model, cost in costs.items() if cost is None)

    def call_cost(models: dict[str, dict[str, int]]) -> float | None:
        values = [model_cost(tokens, prices.get(model)) for model, tokens in models.items()]
        return None if any(value is None for value in values) else sum(values)

    currency = pricing.get("currency", "USD")
    return {
        "token_usage": {
            **_total(by_model),
            "by_model": by_model,
            "by_call_id": {call: _total(models) for call, models in per_call.items()},
            "scope": "chat model tokens, including retried attempts; embeddings not included",
        },
        "estimated_cost": {
            "amount": None if unpriced else sum(costs.values()),
            "currency": currency,
            "by_call_id": {call: call_cost(models) for call, models in per_call.items()},
            "unpriced_models": unpriced,
            "prices_used": {model: {key: value for key, value in price.items() if key != "history"}
                            for model, price in prices.items() if price},
            "unit": pricing.get("unit", "per 1,000,000 tokens"),
        },
        "trace_ids": NOT_RECORDED,
    }
