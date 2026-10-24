"""Model token usage, carried in graph state so every caller's output includes it.

`track_usage` wraps the nodes that call a model. Inside the wrapper LangChain's standard
usage callback totals the tokens of every model call the node makes (all retried attempts
included, since each is a real billed call) and the total is added to the node's update
under `usage`, shaped `{model_name: {input_tokens, output_tokens, total_tokens, ...}}`.
`merge_usage` is the state reducer that sums those per-node totals across the parallel
`check_item` branches, so the graph's final output carries one run total.

This is provider-neutral: any LangChain chat model reports the same usage shape. It records
tokens only. Providers do not return a price, so turning tokens into money is done by the
caller (the evaluation framework does it from a per-model price table).
"""
from __future__ import annotations

import functools
import time
from contextvars import ContextVar
from typing import Any, Callable

from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages.ai import add_usage

_calls: ContextVar[list[dict] | None] = ContextVar("compliance_llm_calls", default=None)
_native_usage: ContextVar[list[dict] | None] = ContextVar("compliance_native_usage", default=None)


def record_native_usage(model: str, usage: dict) -> None:
    """Record direct SDK tokens once, including malformed billed responses."""
    collector = _native_usage.get()
    if collector is not None:
        collector.append({model: usage})


def record_call(record: dict) -> None:
    """Attach safe request diagnostics to the active node, including worker calls."""
    calls = _calls.get()
    if calls is not None:
        calls.append(record)


def merge_usage(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """State reducer: sum two `{model: usage}` mappings."""
    merged = dict(left or {})
    for model, usage in (right or {}).items():
        merged[model] = add_usage(merged[model], usage) if model in merged else usage
    return merged


def merge_stage_usage(left: dict | None, right: dict | None) -> dict:
    merged = dict(left or {})
    for stage, usage in (right or {}).items():
        merged[stage] = merge_usage(merged.get(stage), usage)
    return merged


def track_usage(node: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a node so its model-token usage is added to its state update."""

    @functools.wraps(node)
    def wrapped(state: Any, *args: Any, **kwargs: Any) -> Any:
        calls: list[dict] = []
        native: list[dict] = []
        native_token = _native_usage.set(native)
        token = _calls.set(calls)
        started = time.perf_counter()
        try:
            with get_usage_metadata_callback() as usage:
                update = node(state, *args, **kwargs)
        finally:
            _calls.reset(token)
            _native_usage.reset(native_token)
        batch_usage = {}
        for entry in native:
            batch_usage = merge_usage(batch_usage, entry)
        standard_usage = dict(usage.usage_metadata)
        total = merge_usage(standard_usage, batch_usage)
        if isinstance(update, dict) and total:
            stage = node.__name__.removesuffix("_node")
            transports = {}
            if standard_usage:
                transports["standard"] = standard_usage
            if batch_usage:
                transports["batch"] = batch_usage
            update = {**update, "usage": total, "usage_by_stage": {stage: total},
                      "usage_by_transport": transports}
        if isinstance(update, dict) and calls:
            update = {**update, "llm_calls": [
                {**call, "node": node.__name__.removesuffix("_node"),
                 "item_id": state.get("item_id"),
                 "node_latency_ms": round((time.perf_counter() - started) * 1000, 3)}
                for call in calls
            ]}
        return update

    return wrapped


def estimate_reported_cost(usage_by_transport: dict, price_book: dict) -> dict:
    """Price reported tokens using explicit per-model, per-transport rate snapshots.

    Rates are per million tokens. No default prices or automatic Batch discount:
    callers must supply current standard/batch/cached-input rates and their source.
    Thinking is already in output_tokens. This does not price embeddings or
    unreported usage, and does not change the separate evaluation cost engine.
    """
    amounts, unpriced, used = {}, [], {}
    for transport, models in (usage_by_transport or {}).items():
        if not models:
            continue
        amount = 0.0
        complete = True
        for name, usage in models.items():
            model = name.removeprefix("models/")
            price = price_book.get("models", {}).get(model, {}).get(transport)
            fresh = int(usage.get("input_tokens") or 0)
            output = int(usage.get("output_tokens") or 0)
            cached = int((usage.get("input_token_details") or {}).get("cache_read") or 0)
            if min(fresh, output, cached) < 0 or cached > fresh:
                raise ValueError("Invalid reported token counts")
            keys = ["input", "output"] + (["cached_input"] if cached else [])
            if not price or any(price.get(key) is None or price[key] < 0 for key in keys):
                unpriced.append(f"{transport}/{model}")
                complete = False
                continue
            cache_cost = cached * price["cached_input"] if cached else 0
            amount += ((fresh - cached) * price["input"] + output * price["output"]
                       + cache_cost) / 1_000_000
            used.setdefault(model, {})[transport] = dict(price)
        amounts[transport] = amount if complete else None
    return {
        "amount": sum(amounts.values()) if amounts and not unpriced else None,
        "currency": price_book.get("currency", "USD"),
        "by_transport": amounts, "unpriced_models": sorted(unpriced),
        "prices_used": used, "source": price_book.get("source"), "as_of": price_book.get("as_of"),
        "scope": "reported chat tokens including reported retries; embeddings and unreported usage excluded",
    }
