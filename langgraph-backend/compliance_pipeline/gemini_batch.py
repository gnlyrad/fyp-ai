"""Opt-in native Gemini Batch API transport, distinct from question grouping.

Requests are independent: no transcript or answer is shared between entries.
Batch creation is not idempotent, so an uncertain submission or pending job is
never resubmitted automatically. Safe receipts contain job IDs and token usage,
not prompts. The normal online transport remains the default.
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import uuid
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path

from google.genai import types


class BatchTransportError(RuntimeError):
    no_retry = True

    def __init__(self, reason, job_name=None):
        self.job_name = job_name
        super().__init__(reason)


@dataclass
class _Request:
    model: str
    request: types.InlinedRequest
    deadline: float
    key: str
    future: Future = field(default_factory=Future)


def make_request(system, user, schema, *, max_output_tokens, thinking_level="", seed=42, temperature=1):
    """Keep the same function-calling answer format as online inference."""
    declaration = types.FunctionDeclaration(
        name=schema.__name__, description="Return the complete requested structured answer.",
        parameters_json_schema=schema.model_json_schema(),
    )
    return types.InlinedRequest(
        contents=[types.Content(role="user", parts=[types.Part(text=user)])],
        config=types.GenerateContentConfig(
            system_instruction=system, temperature=temperature, max_output_tokens=max_output_tokens,
            seed=seed, tools=[types.Tool(function_declarations=[declaration])],
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=[schema.__name__])),
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level) if thinking_level else None,
        ),
    )


def parse_response(response, schema):
    """Validate exactly one matching function call; never guess from free text."""
    calls = [part.function_call for candidate in response.candidates or []
             for part in (candidate.content.parts if candidate.content else []) or []
             if part.function_call is not None]
    if len(calls) != 1 or calls[0].name != schema.__name__:
        from compliance_pipeline.reliability import InvalidStructuredOutput
        raise InvalidStructuredOutput("Batch response must contain exactly the requested function")
    return schema.model_validate(calls[0].args)


def response_usage(response):
    """Match LangChain's token shape. Thinking is included in output once."""
    usage = response.usage_metadata
    if usage is None:
        return None
    prompt = usage.prompt_token_count or 0
    reasoning = usage.thoughts_token_count or 0
    output = (usage.candidates_token_count or 0) + reasoning
    return {
        "input_tokens": prompt, "output_tokens": output,
        "total_tokens": usage.total_token_count or prompt + output,
        "input_token_details": {"cache_read": usage.cached_content_token_count or 0},
        "output_token_details": {"reasoning": reasoning},
    }


class NativeBatchTransport:
    """Coalesce concurrent generation requests, then poll a bounded number of jobs.

    Waiting is intentional for this asynchronous, non-interactive provider mode.
    Timeouts return a pending-job error, not a second billed job or online fallback.
    """

    def __init__(self, client, receipts, *, max_requests=20, gather_seconds=0.1,
                 poll_seconds=10, max_wait_seconds=86400, max_jobs=4):
        if not 1 <= max_requests <= 100 or not 1 <= max_jobs <= 8:
            raise ValueError("Invalid native batch size or concurrent-job limit")
        if not 0 <= gather_seconds <= 5 or not 0 < poll_seconds <= 60 or not 0 < max_wait_seconds <= 86400:
            raise ValueError("Invalid native batch timing")
        self.client, self.receipts = client, Path(receipts)
        self.max_requests, self.gather_seconds = max_requests, gather_seconds
        self.poll_seconds, self.max_wait_seconds = poll_seconds, max_wait_seconds
        self.pending = queue.Queue(maxsize=1000)
        self.slots = threading.Semaphore(max_jobs)
        self.stopping = threading.Event()
        self.worker = threading.Thread(target=self._dispatch, daemon=True)
        self.worker.start()

    def invoke(self, model, request):
        if self.stopping.is_set():
            raise BatchTransportError("Batch transport is closed")
        key = uuid.uuid4().hex
        request = request.model_copy(deep=True, update={"metadata": {"request_id": key}})
        # Inline requests are limited to 20 MB. Leave room for JSON framing.
        if len(request.model_dump_json(exclude_none=True).encode("utf-8")) > 18_000_000:
            raise BatchTransportError("Request exceeds the inline Batch API size limit")
        entry = _Request(model, request, time.monotonic() + self.max_wait_seconds, key)
        try:
            self.pending.put_nowait(entry)
        except queue.Full as exc:
            raise BatchTransportError("Native batch queue is full") from exc
        try:
            return entry.future.result(timeout=self.max_wait_seconds + 1)
        except FutureTimeout as exc:
            entry.future.cancel()
            raise BatchTransportError("Batch wait expired; inspect job receipts before another submission") from exc

    def close(self):
        self.stopping.set()
        # The worker drains pending local requests without submitting them.
        self.worker.join(timeout=1)
        close_client = getattr(self.client, "close", None)
        if callable(close_client):
            close_client()

    @staticmethod
    def _fail(entries, error):
        for entry in entries:
            if not entry.future.done():
                entry.future.set_exception(error)

    def _dispatch(self):
        while not self.stopping.is_set():
            try:
                first = self.pending.get(timeout=0.1)
            except queue.Empty:
                continue
            entries = [first]
            deadline = time.monotonic() + self.gather_seconds
            size = len(first.request.model_dump_json(exclude_none=True).encode("utf-8"))
            while len(entries) < self.max_requests and time.monotonic() < deadline:
                try:
                    entry = self.pending.get(timeout=max(0.001, deadline - time.monotonic()))
                except queue.Empty:
                    break
                entry_size = len(entry.request.model_dump_json(exclude_none=True).encode("utf-8"))
                if size + entry_size > 18_000_000:
                    self._start([entry])
                    break
                entries.append(entry)
                size += entry_size
            # A provider job uses one model; generation settings remain per request.
            for model in dict.fromkeys(entry.model for entry in entries):
                self._start([entry for entry in entries if entry.model == model])
        while True:
            try:
                self._fail([self.pending.get_nowait()], BatchTransportError("Batch transport closed before submission"))
            except queue.Empty:
                break

    def _start(self, entries):
        deadline = min(entry.deadline for entry in entries)
        while not self.slots.acquire(timeout=0.05):
            if self.stopping.is_set() or time.monotonic() >= deadline:
                self._fail(entries, BatchTransportError("Batch queue wait expired before submission"))
                return
        threading.Thread(target=self._run_job, args=(entries,), daemon=True).start()

    def _run_job(self, entries):
        job_name = None
        record = None
        path = None

        def save():
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
            temporary.replace(path)

        try:
            active = []
            for entry in entries:
                if (not self.stopping.is_set() and time.monotonic() < entry.deadline
                        and entry.future.set_running_or_notify_cancel()):
                    active.append(entry)
                elif not entry.future.done():
                    self._fail([entry], BatchTransportError("Batch expired before submission"))
            entries = active
            if not entries:
                return
            self.receipts.mkdir(parents=True, exist_ok=True)
            path = self.receipts / (uuid.uuid4().hex + ".json")
            record = {
                "transport": "batch", "state": "submitting", "model": entries[0].model,
                "request_ids": [entry.key for entry in entries],
                "request_sha256": [hashlib.sha256(entry.request.model_dump_json(
                    exclude_none=True).encode()).hexdigest() for entry in entries],
            }
            # Do not submit if a durable, safe receipt cannot be created.
            with path.open("x", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
            try:
                job = self.client.batches.create(
                    model=entries[0].model, src=[entry.request for entry in entries],
                    config={"display_name": "production-nlp-" + path.stem},
                )
            except Exception as exc:
                record["state"] = "submission_uncertain"
                record["error_type"] = type(exc).__name__
                code = getattr(exc, "code", None)
                if isinstance(code, int):
                    record["http_code"] = code
                    if 400 <= code < 500 and code != 408:
                        record["state"] = "provider_rejected"
                save()
                raise BatchTransportError("Batch submission failed or is uncertain; do not resubmit automatically") from exc
            job_name = job.name
            if not job_name:
                raise BatchTransportError("Provider did not return a batch job name")
            record.update(job_name=job_name, state="submitted")
            save()
            completed = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED",
                         "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
            while True:
                state = job.state.value if hasattr(job.state, "value") else str(job.state)
                if state in completed:
                    break
                left = min(entry.deadline for entry in entries) - time.monotonic()
                if left <= 0 or self.stopping.wait(min(self.poll_seconds, max(0, left))):
                    record["state"] = "pending_after_wait"
                    save()
                    raise BatchTransportError("Native batch is still pending; inspect the saved job", job_name)
                job = self.client.batches.get(name=job_name)
            record["state"] = state
            save()
            if state not in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
                raise BatchTransportError("Native batch did not succeed", job_name)
            rows = list(job.dest.inlined_responses or []) if job.dest else []
            record["usage_by_response"] = [
                {"request_id": (row.metadata or {}).get("request_id"),
                 "usage": response_usage(row.response) if row.response else None} for row in rows]
            save()
            expected = {entry.key for entry in entries}
            ids = [(row.metadata or {}).get("request_id") for row in rows]
            if len(ids) != len(expected) or set(ids) != expected:
                raise BatchTransportError("Batch response IDs are missing, duplicated or unexpected", job_name)
            by_id = dict(zip(ids, rows))
            record["usage_by_request"] = {
                key: response_usage(row.response) if row.response else None for key, row in by_id.items()}
            record["failed_request_ids"] = [key for key, row in by_id.items() if row.error or not row.response]
            save()
            for entry in entries:
                if not entry.future.done():
                    entry.future.set_result((by_id[entry.key], job_name))
        except Exception as exc:
            error = exc if isinstance(exc, BatchTransportError) else BatchTransportError(
                "Native batch unavailable; inspect receipts before resubmitting", job_name)
            self._fail(entries, error)
        finally:
            self.slots.release()
