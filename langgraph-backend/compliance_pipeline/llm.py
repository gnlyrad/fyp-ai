"""The single entry point every node uses to call an LLM.

Nodes never call a chat model directly (CLAUDE.md "Rules for working in this
repo"). This keeps a later model swap to one file, and guarantees the graph
never crashes on a bad or missing LLM response: `get_llm_response` retries
once and, on a second failure, returns `LLMResult(ok=False, ...)` instead of
raising. Callers turn that into a `CANNOT_DETERMINE` result plus an `errors`
entry.

Gemini only for this package (per plan); `compliance_pipeline/model.py`'s
Bedrock/Nova default is out of scope here.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel

load_dotenv(find_dotenv())  # must run before the client reads GOOGLE_API_KEY

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL_ID = "gemini-3.6-flash"

# Gemini's default `json_schema` structured-output mode drops nested models
# (e.g. a list of observations each holding evidence quotes): it returns an
# empty list and runs the free-text field on for thousands of characters.
# Tool-calling mode fills nested schemas correctly, so it is the default here.
STRUCTURED_METHOD = os.getenv("COMPLIANCE_STRUCTURED_METHOD", "function_calling")
# Bound every call so a runaway generation fails (and is retried) instead of
# hanging the graph.
MAX_OUTPUT_TOKENS = int(os.getenv("COMPLIANCE_MAX_OUTPUT_TOKENS", "4096"))
REQUEST_TIMEOUT_S = float(os.getenv("COMPLIANCE_LLM_TIMEOUT_S", "90"))
# Fixed sampling seed so re-running a call gives the same judgement
# (gemini-3.6-flash ignores temperature). COMPLIANCE_LLM_SEED=none disables it.
_SEED = os.getenv("COMPLIANCE_LLM_SEED", "42").strip().lower()
SEED = None if _SEED in ("", "none") else int(_SEED)

# Backend signature: (prompt_name, system, user, schema, model) -> BaseModel.
# Raises on failure; get_llm_response() is what catches and retries.
Backend = Callable[[str, str, str, type[BaseModel], str], BaseModel]


@dataclass
class LLMResult(Generic[T]):
    ok: bool
    value: T | None = None
    error: str | None = None
    attempts: int = 0


_clients: dict[str, object] = {}  # ChatGoogleGenerativeAI per model id, built lazily


def _get_client(model_id: str):
    if model_id not in _clients:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _clients[model_id] = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            timeout=REQUEST_TIMEOUT_S,
            max_retries=0,  # get_llm_response() owns the retry
            seed=SEED,
        )
    return _clients[model_id]


def _real_backend(prompt_name: str, system: str, user: str, schema: type[BaseModel], model: str) -> BaseModel:
    """Default backend: Gemini via `with_structured_output`. `model` overrides the default id if set."""
    client = _get_client(model or os.getenv("COMPLIANCE_MODEL_ID", DEFAULT_MODEL_ID))
    structured = client.with_structured_output(schema, method=STRUCTURED_METHOD)
    messages = [("system", system), ("human", user)]
    result = structured.invoke(messages)
    if result is None:  # tool-calling mode returns None when the model skips the tool call
        raise ValueError("model returned no structured output")
    return result


_backend: Backend = _real_backend


def set_backend(fn: Backend) -> None:
    """Swap the backend, e.g. in tests with a fake that returns canned responses."""
    global _backend
    _backend = fn


def reset_backend() -> None:
    """Restore the real Gemini backend."""
    global _backend
    _backend = _real_backend


# Provider-side overload / rate-limit / timeout errors are worth waiting out;
# anything else (bad output, validation) gets the single CLAUDE.md retry.
_TRANSIENT_RE = re.compile(
    r"\b(429|500|502|503|504)\b|UNAVAILABLE|RESOURCE_EXHAUSTED|DEADLINE_EXCEEDED|"
    r"overloaded|high demand|timed? ?out|timeout",
    re.IGNORECASE,
)
# Waits (seconds) before each extra attempt after a transient error.
TRANSIENT_BACKOFF_S = [
    float(x) for x in os.getenv("COMPLIANCE_LLM_BACKOFF_S", "5,15,30,60").split(",") if x.strip()
]


def _is_transient(exc: Exception) -> bool:
    return bool(_TRANSIENT_RE.search(f"{type(exc).__name__}: {exc}"))


def get_llm_response(
    prompt_name: str,
    system: str,
    user: str,
    schema: type[T],
    *,
    model: str | None = None,
) -> LLMResult[T]:
    """Call the current backend and return validated output. Never raises.

    A bad or invalid response is retried once (CLAUDE.md rule). A transient
    provider error (overload, rate limit, timeout) is retried after each wait
    in `TRANSIENT_BACKOFF_S`, without using up that single retry. When both
    budgets are spent, returns `LLMResult(ok=False, error=...)`.
    """
    last_error: str | None = None
    attempts = 0
    output_retries_left = 1
    backoff = list(TRANSIENT_BACKOFF_S)
    while True:
        attempts += 1
        try:
            value = _backend(prompt_name, system, user, schema, model or "")
            if not isinstance(value, schema):
                value = schema.model_validate(value)
            return LLMResult(ok=True, value=value, attempts=attempts)
        except Exception as exc:  # noqa: BLE001 - covers ValidationError; never raise out of here
            last_error = f"{type(exc).__name__}: {exc}"
            if _is_transient(exc) and backoff:
                time.sleep(backoff.pop(0))
                continue
            if output_retries_left:
                output_retries_left -= 1
                continue
            return LLMResult(ok=False, error=last_error, attempts=attempts)
