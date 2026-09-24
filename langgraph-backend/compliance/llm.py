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
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel

load_dotenv(find_dotenv())  # must run before the client reads GOOGLE_API_KEY

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL_ID = "gemini-2.5-flash"

# Backend signature: (prompt_name, system, user, schema, model) -> BaseModel.
# Raises on failure; get_llm_response() is what catches and retries.
Backend = Callable[[str, str, str, type[BaseModel], str], BaseModel]


@dataclass
class LLMResult(Generic[T]):
    ok: bool
    value: T | None = None
    error: str | None = None
    attempts: int = 0


_client = None  # lazily-built ChatGoogleGenerativeAI, one per process


def _get_client():
    global _client
    if _client is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_id = os.getenv("COMPLIANCE_MODEL_ID", DEFAULT_MODEL_ID)
        _client = ChatGoogleGenerativeAI(model=model_id, temperature=0)
    return _client


def _real_backend(prompt_name: str, system: str, user: str, schema: type[BaseModel], model: str) -> BaseModel:
    """Default backend: Gemini via `with_structured_output`. `model` overrides the client's default id if set."""
    client = _get_client()
    if model:
        from langchain_google_genai import ChatGoogleGenerativeAI

        client = ChatGoogleGenerativeAI(model=model, temperature=0)
    structured = client.with_structured_output(schema)
    messages = [("system", system), ("human", user)]
    return structured.invoke(messages)


_backend: Backend = _real_backend


def set_backend(fn: Backend) -> None:
    """Swap the backend, e.g. in tests with a fake that returns canned responses."""
    global _backend
    _backend = fn


def reset_backend() -> None:
    """Restore the real Gemini backend."""
    global _backend
    _backend = _real_backend


def get_llm_response(
    prompt_name: str,
    system: str,
    user: str,
    schema: type[T],
    *,
    model: str | None = None,
) -> LLMResult[T]:
    """Call the current backend, retrying once on any exception or validation error.

    Never raises. On two failures, returns `LLMResult(ok=False, error=...)`.
    """
    last_error: str | None = None
    for attempt in (1, 2):
        try:
            value = _backend(prompt_name, system, user, schema, model or "")
            if not isinstance(value, schema):
                value = schema.model_validate(value)
            return LLMResult(ok=True, value=value, attempts=attempt)
        except Exception as exc:  # noqa: BLE001 - covers ValidationError; never raise out of here
            last_error = f"{type(exc).__name__}: {exc}"
    return LLMResult(ok=False, error=last_error, attempts=2)
