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
        token = _calls.set(calls)
        started = time.perf_counter()
        try:
            with get_usage_metadata_callback() as usage:
                update = node(state, *args, **kwargs)
        finally:
            _calls.reset(token)
        if isinstance(update, dict) and usage.usage_metadata:
            stage = node.__name__.removesuffix("_node")
            update = {**update, "usage": dict(usage.usage_metadata),
                      "usage_by_stage": {stage: dict(usage.usage_metadata)}}
        if isinstance(update, dict) and calls:
            update = {**update, "llm_calls": [
                {**call, "node": node.__name__.removesuffix("_node"),
                 "item_id": state.get("item_id"),
                 "node_latency_ms": round((time.perf_counter() - started) * 1000, 3)}
                for call in calls
            ]}
        return update

    return wrapped
