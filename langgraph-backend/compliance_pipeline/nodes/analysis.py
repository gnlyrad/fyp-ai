"""Analysis node: one LLM judgement for one checklist item.

Fanned out with `Send`, so this function runs once per checklist item and
returns a single-element `findings` list. The `operator.add` reducer on
`ComplianceState.findings` concatenates them.
"""
from __future__ import annotations

import os
from typing import Any

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..prompts.checklist import ANALYSIS_SYSTEM_PROMPT
from ..state import AnalysisTask, Finding
from .retrieval import build_context, format_timestamp

load_dotenv(find_dotenv())  # must run before init_chat_model()

MODEL_ID = os.getenv("CLEARCALL_MODEL_ID", "apac.amazon.nova-lite-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")


class Judgement(BaseModel):
    """Schema the model is forced to fill in."""

    verdict: str = Field(description="One of: Pass, Fail, Unclear, N/A")
    reason: str = Field(description="One sentence justifying the verdict.")
    evidence_quote: str = Field(
        default="",
        description="Text copied verbatim from the excerpts, or empty if unsupported.",
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


def analysis_node(task: AnalysisTask) -> dict[str, Any]:
    """Judge one checklist item against its retrieved evidence."""
    item = task["item"]
    retrieved = task["retrieved"]
    context = build_context(retrieved)

    try:
        answer = judge.invoke([
            SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(content=f"Compliance question: {item['question']}\n\nTranscript excerpts:\n{context}"),
        ])
        verdict = normalise_verdict(answer.verdict)
        quote = (answer.evidence_quote or "").strip()
        reason = answer.reason
        error = None
    except Exception as exc:  # model unavailable, throttled, or malformed output
        verdict, quote, reason, error = None, "", str(exc), exc

    # Relevance gate: the cited quote must actually occur in retrieved transcript
    # text. This is what stops a fluent but unsupported "Pass" from getting through.
    matched = next(
        (r for r in retrieved if quote and quote.lower() in r["text"].lower()),
        None,
    )
    supported = matched is not None

    if error is not None:
        verdict = "Unclear"
        reason = f"Analysis unavailable: {reason}"
    elif not verdict or not supported:
        verdict = "Unclear"
        reason = "Insufficient retrieved evidence to support a definitive outcome."

    finding: Finding = {
        "checklist_id": item["id"],
        "question": item["question"],
        "verdict": verdict,
        "reason": reason,
        "confidence": "high" if supported and verdict not in {"Unclear", "N/A"} else "low",
        "evidence_quote": quote if supported else None,
        "timestamp": format_timestamp(matched["start_seconds"]) if supported else None,
        "chunk_id": matched["chunk_id"] if supported else None,
        "retrieved_evidence": retrieved,
    }
    return {"findings": [finding]}
