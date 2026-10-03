"""Bounded, per-stage recovery for model transport and structured-output errors."""
from __future__ import annotations

import json
import os
import random
import time

from botocore.exceptions import ClientError, ConnectionError as AWSConnectionError, HTTPClientError
from google.genai.errors import APIError as GoogleAPIError
from httpx import TransportError
from langchain_core.exceptions import OutputParserException
from langchain.messages import HumanMessage
from pydantic import ValidationError


class InvalidStructuredOutput(ValueError):
    """A parsed response is missing required semantic structure."""


class StageFailure(Exception):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(f"{detail['stage']} unavailable: {detail['error_type']}")


def analysis_options(values):
    mode = values.get("context_mode", os.getenv("CLEARCALL_CONTEXT_MODE", "routed"))
    attempts = values.get("analysis_max_attempts", os.getenv("CLEARCALL_ANALYSIS_MAX_ATTEMPTS", "3"))
    if mode not in {"routed", "full_transcript"}:
        raise ValueError("context_mode must be routed or full_transcript")
    if isinstance(attempts, bool) or str(attempts) not in {"1", "2", "3", "4", "5"}:
        raise ValueError("analysis_max_attempts must be an integer from 1 to 5")
    return {"context_mode": mode, "analysis_max_attempts": int(attempts)}


def error_details(exc):
    code = None
    transient = isinstance(exc, (TimeoutError, ConnectionError, AWSConnectionError, HTTPClientError, TransportError))
    # LangChain may wrap Google's error. Inspect causes without exposing messages.
    cause, seen = exc, set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, GoogleAPIError):
            code = str(cause.code)
            transient = cause.code in {408, 429, 500, 502, 503, 504}
            break
        cause = cause.__cause__
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code")
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        transient = code in {
            "ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException",
            "InternalServerException", "InternalFailure", "ModelTimeoutException",
            "ModelNotReadyException", "RequestTimeout", "RequestTimeoutException",
        } or status in {429, 500, 502, 503, 504}
        if code in {"UnrecognizedClientException", "InvalidSignatureException", "ExpiredTokenException",
                    "AccessDeniedException", "ValidationException", "ResourceNotFoundException"}:
            transient = False
    malformed = isinstance(exc, (ValidationError, OutputParserException, json.JSONDecodeError,
                                 InvalidStructuredOutput))
    return {"error_type": type(exc).__name__, "error_code": code,
            "retryable": transient or malformed}, malformed


def invoke_structured(factory, messages, schema, *, stage, diagnostics, max_attempts, validate=None):
    """Retry this stage only. Never retry a valid semantic Unclear response.

    At most one malformed-output repair is allowed, within the total attempt
    budget. Do not persist raw exception messages, which can contain input data.
    Counts describe application invocations, not the provider SDK's HTTP retries.
    """
    record = {"stage": stage, "attempts": 0, "status": "running", "recovered": False, "errors": []}
    diagnostics.append(record)
    repair_used = False
    current_messages = list(messages)
    for attempt in range(1, max_attempts + 1):
        record["attempts"] = attempt
        try:
            answer = schema.model_validate(factory().invoke(current_messages))
            if validate:
                validate(answer)
            record.update(status="success", recovered=bool(record["errors"]))
            return answer
        except Exception as exc:
            detail, malformed = error_details(exc)
            retry = detail["retryable"] and attempt < max_attempts and not (malformed and repair_used)
            record["errors"].append({**detail, "attempt": attempt, "will_retry": retry})
            if not retry:
                record["status"] = "prediction_failure"
                raise StageFailure({**detail, "stage": stage, "attempts": attempt}) from exc
            if malformed:
                repair_used = True
                # Feedback contains validation structure only, never raw model output.
                feedback = (str(exc) if isinstance(exc, InvalidStructuredOutput)
                            else "The response did not validate against the required schema.")
                current_messages = [*messages, HumanMessage(content=(
                    f"Output validation failed: {feedback} Return a complete response matching "
                    "the required schema. Preserve the evidence rules; Unclear is allowed."))]
            else:
                time.sleep(random.uniform(0, min(8.0, 2 ** (attempt - 1))))
