"""The single entry point every node uses to call an LLM.

Nodes never call a chat model directly (CLAUDE.md "Rules for working in this
repo"). This keeps a later model swap to one file, and guarantees the graph
never crashes on a bad or missing LLM response: `get_llm_response` retries
once and, on a second failure, returns `LLMResult(ok=False, ...)` instead of
raising. Callers turn that into a `CANNOT_DETERMINE` result plus an `errors`
entry.

Uses the shared Gemini Developer API client and repository-root credentials.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from langchain_backend.models import create_chat_model
from langchain_backend.settings import settings
from compliance_pipeline.reliability import InvalidStructuredOutput, error_details
from compliance_pipeline.usage import record_call
from pydantic import BaseModel


class _DerefedDefsFilter(logging.Filter):
    """Drop one harmless warning from langchain-google-genai.

    It inlines every `$ref` of a schema (so nested models are sent in full) and then logs that
    the now-unused `$defs` key is ignored, once per model call. Only that exact message is
    dropped; every other warning from the library still shows.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Key '$defs' is not supported in schema" not in record.getMessage()


logging.getLogger("langchain_google_genai._function_utils").addFilter(_DerefedDefsFilter())

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL_ID = settings.chat_model

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


_clients: dict[tuple, object] = {}


def _get_client(model_id: str, prompt_name: str = ""):
    prefix = f"COMPLIANCE_{prompt_name.upper()}"
    limit = int(os.getenv(f"{prefix}_MAX_OUTPUT_TOKENS", str(MAX_OUTPUT_TOKENS)))
    default_thinking = "medium" if prompt_name == "decide_actions_repair" else ""
    thinking = os.getenv(f"{prefix}_THINKING_LEVEL", os.getenv("COMPLIANCE_THINKING_LEVEL", default_thinking)).strip()
    if limit < 1 or thinking not in {"", "minimal", "low", "medium", "high"}:
        raise ValueError("Invalid model output limit or thinking level")
    key = (model_id, limit, thinking, REQUEST_TIMEOUT_S, SEED)
    if key not in _clients:
        _clients[key] = create_chat_model(
            model_id=model_id,
            max_output_tokens=limit,
            timeout=REQUEST_TIMEOUT_S,
            max_retries=0,  # get_llm_response() owns the retry
            seed=SEED,
            **({"thinking_level": thinking} if thinking else {}),
        )
    return _clients[key]


def _real_backend(prompt_name: str, system: str, user: str, schema: type[BaseModel], model: str) -> BaseModel:
    """Default backend: Gemini via `with_structured_output`. `model` overrides the default id if set."""
    client = _get_client(model or os.getenv("COMPLIANCE_MODEL_ID", DEFAULT_MODEL_ID), prompt_name)
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
    validate: Callable[[T], None] | None = None,
) -> LLMResult[T]:
    """Run a validated request with stage-specific model settings and safe diagnostics."""
    model_id = (model or os.getenv(f"COMPLIANCE_{prompt_name.upper()}_MODEL_ID")
                or os.getenv("COMPLIANCE_MODEL_ID", DEFAULT_MODEL_ID))
    started = time.perf_counter()
    result = _invoke_response(prompt_name, system, user, schema, model_id, validate)
    record_call({"stage": prompt_name, "model": model_id, "attempts": result.attempts,
                 "status": "success" if result.ok else "prediction_failure", "error": result.error,
                 "latency_ms": round((time.perf_counter() - started) * 1000, 3)})
    return result


def _invoke_response(prompt_name, system, user, schema, model, validate):
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
    max_attempts = int(os.getenv("COMPLIANCE_LLM_MAX_ATTEMPTS", str(len(backoff) + 2)))
    current_user = user
    while True:
        attempts += 1
        try:
            value = _backend(prompt_name, system, current_user, schema, model)
            if not isinstance(value, schema):
                value = schema.model_validate(value)
            if validate:
                validate(value)
            return LLMResult(ok=True, value=value, attempts=attempts)
        except Exception as exc:  # noqa: BLE001 - covers ValidationError; never raise out of here
            detail, malformed = error_details(exc)
            last_error = detail["error_type"]
            if detail["error_code"]:
                last_error += f" (code {detail['error_code']})"
            # Keep provider messages (which may contain credentials/input) out of artifacts.
            if attempts >= max_attempts or (detail["error_code"] and not detail["retryable"]):
                return LLMResult(ok=False, error=last_error, attempts=attempts)
            transient = detail["retryable"] and not malformed or _is_transient(exc)
            if transient and backoff:
                time.sleep(backoff.pop(0))
                continue
            if transient:
                return LLMResult(ok=False, error=last_error, attempts=attempts)
            if output_retries_left:
                output_retries_left -= 1
                feedback = ("\nOutput validation failed. Return the complete requested structure, "
                            "with every requested identifier exactly once and no extra identifiers. "
                            "Preserve the evidence rules; CANNOT_DETERMINE is allowed.")
                current_user = user + feedback
                continue
            return LLMResult(ok=False, error=last_error, attempts=attempts)
