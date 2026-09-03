"""Analysis node: one LLM judgement for one checklist item.

Fanned out with `Send`, so this function runs once per checklist item and
returns a single-element `findings` list plus its verified spans. The
`operator.add` reducers on `ComplianceState` concatenate them.

The model is asked for verbatim quotes; this module is what turns a quote into
an exact `(start, end)` location in the caller's transcript. A quote that cannot
be located is discarded — that is the gate that stops a fluent but unsupported
verdict from being reported as fact.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..prompts.checklist import ANALYSIS_SYSTEM_PROMPT
from ..state import AnalysisTask, Finding, RetrievedChunk, Span
from .retrieval import build_context, chunk_timestamp

load_dotenv(find_dotenv())  # must run before init_chat_model()

MODEL_ID = os.getenv("CLEARCALL_MODEL_ID", "apac.amazon.nova-lite-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")


class EvidenceSpan(BaseModel):
    """One piece of cited evidence."""

    quote: str = Field(description="Text copied verbatim from the excerpts.")
    label: str = Field(description="Either 'compliant' or 'violation'.")


class Judgement(BaseModel):
    """Schema the model is forced to fill in."""

    verdict: str = Field(description="One of: Pass, Fail, Unclear, N/A")
    reason: str = Field(description="One sentence justifying the verdict.")
    evidence: list[EvidenceSpan] = Field(
        default_factory=list,
        description="Every relevant verbatim quote, each labelled compliant or violation. "
                    "Empty if nothing in the excerpts supports an answer.",
    )


_model = init_chat_model(
    MODEL_ID,
    model_provider="bedrock_converse",
    region_name=AWS_REGION,
    temperature=0,
)
judge = _model.with_structured_output(Judgement)


def normalise_verdict(value: str) -> str | None:
    choices = {
        "pass": "Pass", "fail": "Fail", "unclear": "Unclear",
        "n/a": "N/A", "na": "N/A", "not applicable": "N/A",
    }
    return choices.get((value or "").strip().lower())


def normalise_label(value: str) -> str:
    """Anything that is not clearly a violation is treated as compliant evidence."""
    text = (value or "").strip().lower()
    if text in {"violation", "violating", "non-compliant", "noncompliant", "fail", "breach"}:
        return "violation"
    return "compliant"


def locate(
    quote: str,
    retrieved: list[RetrievedChunk],
    transcript: str,
    checklist_id: str,
) -> Span | None:
    """Resolve a model-supplied quote to exact character offsets in `transcript`.

    Searched within the retrieved chunks only, so a quote lifted from elsewhere in
    the call (or invented) cannot be attributed to this item's evidence. The final
    slice check is what guarantees `transcript[start:end] == text` for the caller.
    """
    quote = (quote or "").strip()
    if not quote:
        return None

    needle = quote.lower()
    for chunk in retrieved:
        offset = chunk["text"].lower().find(needle)
        if offset == -1:
            continue
        start = chunk["start_char"] + offset
        end = start + len(quote)
        if transcript[start:end].lower() != needle:
            continue  # chunk text drifted from the transcript; refuse to guess
        return {
            "text": transcript[start:end],
            "start": start,
            "end": end,
            "label": "compliant",  # overwritten by the caller
            "checklist_id": checklist_id,
            "chunk_id": chunk["chunk_id"],
            "timestamp": chunk_timestamp(chunk),
        }
    return None


def analysis_node(task: AnalysisTask) -> dict[str, Any]:
    """Judge one checklist item against its retrieved evidence."""
    item = task["item"]
    retrieved = task["retrieved"]
    transcript = task["transcript"]
    context = build_context(retrieved)


    try:
        answer = judge.invoke([
            SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=f"Compliance question: {item['question']}\n\nTranscript excerpts:\n{context}"),
        ])
        verdict = normalise_verdict(answer.verdict)
        cited = answer.evidence or []
        reason = answer.reason
        error = None
    except Exception as exc:  # model unavailable, throttled, or malformed output
        verdict, cited, reason, error = None, [], str(exc), exc

    # Relevance gate: every cited quote must actually occur in the retrieved
    # transcript text. Unlocatable quotes are dropped rather than reported.
    spans: list[Span] = []
    for evidence in cited:
        span = locate(evidence.quote, retrieved, transcript, item["id"])
        if span is not None:
            span["label"] = normalise_label(evidence.label)
            spans.append(span)

    supported = bool(spans)

    if error is not None:
        verdict = "Unclear"
        reason = f"Analysis unavailable: {reason}"
    elif not verdict or not supported:
        verdict = "Unclear"
        reason = "Insufficient retrieved evidence to support a definitive outcome."

    first = spans[0] if spans else None
    finding: Finding = {
        "checklist_id": item["id"],
        "question": item["question"],
        "verdict": verdict,
        "reason": reason,
        "confidence": "high" if supported and verdict not in {"Unclear", "N/A"} else "low",
        "spans": spans,
        "evidence_quote": first["text"] if first else None,
        "timestamp": first["timestamp"] if first else None,
        "chunk_id": first["chunk_id"] if first else None,
        "retrieved_evidence": retrieved,
    }
    return {"findings": [finding], "spans": spans}
