"""Analysis nodes: one judgement for one checklist item.

There are four of them, one per question type, and the graph routes each item to
the matching node. They differ only in the system prompt they send (and, for
audio metrics, in not calling the model at all) — every one of them runs the
same `_run_analysis` core, so the quote-verification gate below is defined once
and cannot drift between types.

Fanned out with `Send`, so exactly one of these runs per checklist item and
returns a single-element `findings` list plus its verified spans. The
`operator.add` reducers on `ComplianceState` concatenate them.

The model is asked for verbatim quotes; this module is what turns a quote into
an exact `(start, end)` location in the caller's transcript. A quote that cannot
be located is discarded — that is the gate that stops a fluent but unsupported
verdict from being reported as fact.
"""
from __future__ import annotations

from typing import Any

from langchain.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..model import get_model
from ..prompts.analysis import (
    DETERMINISTIC_SYSTEM_PROMPT,
    EXHAUSTIVE_SYSTEM_PROMPT,
    TARGETED_SYSTEM_PROMPT,
)
from ..state import AnalysisTask, Finding, RetrievedChunk, Span
from .retrieval import build_context


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


_judge = None


def judge_model():
    """The structured-output judge, built on first use and shared by all types."""
    global _judge
    if _judge is None:
        _judge = get_model().with_structured_output(Judgement)
    return _judge


def normalise_verdict(value: str) -> str | None:
    choices = {
        "pass": "Pass", "fail": "Fail", "unclear": "Unclear",
        "n/a": "N/A", "na": "N/A", "not applicable": "N/A",
    }
    return choices.get((value or "").strip().lower())


# A finding's evidence must agree with its verdict: a Fail cannot be evidenced
# solely by passages labelled "compliant", which is what the model returned for
# `cooling_off` (verdict Fail, both spans labelled compliant).
VERDICT_LABEL: dict[str, str] = {"Pass": "compliant", "Fail": "violation"}


def normalise_label(value: str) -> str | None:
    """Map the model's label onto our vocabulary; None when it says nothing usable."""
    text = (value or "").strip().lower()
    if text in {"violation", "violating", "non-compliant", "noncompliant", "fail", "breach"}:
        return "violation"
    if text in {"compliant", "compliance", "complying", "pass", "met"}:
        return "compliant"
    return None


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
        }
    return None


def build_finding(
    item: dict,
    retrieved: list[RetrievedChunk],
    spans: list[Span],
    verdict: str,
    reason: str,
) -> Finding:
    """Assemble the finding once the verdict and its verified spans are final."""
    supported = bool(spans)
    first = spans[0] if spans else None
    return {
        "checklist_id": item["id"],
        "question": item["question"],
        "question_type": item.get("question_type", "targeted"),
        "verdict": verdict,
        "reason": reason,
        "confidence": "high" if supported and verdict not in {"Unclear", "N/A"} else "low",
        "spans": spans,
        "evidence_quote": first["text"] if first else None,
        "chunk_id": first["chunk_id"] if first else None,
        "retrieved_evidence": retrieved,
    }


def _run_analysis(task: AnalysisTask, system_prompt: str) -> dict[str, Any]:
    """Judge one checklist item against its retrieved evidence.

    The prompt varies by question type; everything after the model call — quote
    location, verdict normalisation, label reconciliation — does not, and is the
    part that makes the output auditable.
    """
    item = task["item"]
    retrieved = task["retrieved"]
    transcript = task["transcript"]
    context = build_context(retrieved)

    try:
        answer = judge_model().invoke([
            SystemMessage(content=system_prompt),
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
            span["model_label"] = normalise_label(evidence.label)
            spans.append(span)

    supported = bool(spans)

    if error is not None:
        verdict = "Unclear"
        reason = f"Analysis unavailable: {reason}"
    elif not verdict or not supported:
        verdict = "Unclear"
        reason = "Insufficient retrieved evidence to support a definitive outcome."

    # Labels are assigned only once the verdict is final — including after any
    # downgrade above — so a span can never contradict the finding it supports.
    expected = VERDICT_LABEL.get(verdict)
    for span in spans:
        model_label = span.get("model_label")
        span["label"] = expected or model_label or "compliant"
        span["label_corrected"] = expected is not None and model_label != expected

    return {
        "findings": [build_finding(item, retrieved, spans, verdict, reason)],
        "spans": spans,
    }


def analysis_targeted_node(task: AnalysisTask) -> dict[str, Any]:
    """One specific passage, judged on quality as well as presence."""
    return _run_analysis(task, TARGETED_SYSTEM_PROMPT)


def analysis_deterministic_node(task: AnalysisTask) -> dict[str, Any]:
    """A presence check, over the wider evidence set retrieval gathered for it.

    The mechanical half of this check already ran in `retrieval`: every chunk
    literally containing one of the item's terms is in `retrieved`, so silence
    here really is absence, which is why the prompt turns it into Fail rather
    than Unclear.
    """
    return _run_analysis(task, DETERMINISTIC_SYSTEM_PROMPT)


def analysis_exhaustive_node(task: AnalysisTask) -> dict[str, Any]:
    """Every occurrence, over the whole transcript."""
    return _run_analysis(task, EXHAUSTIVE_SYSTEM_PROMPT)


AUDIO_METRICS_UNAVAILABLE = (
    "Audio metrics unavailable: the pipeline is transcript-only. Answering this "
    "needs speaker diarisation and per-turn timings from AWS Transcribe."
)


def analysis_audio_node(task: AnalysisTask) -> dict[str, Any]:
    """Talk-over, silence, and pace — not answerable from text, so not guessed.

    No model call: there is nothing in a transcript for it to read. Reported as
    N/A with the reason stated, rather than as an Unclear that looks like the
    model tried and failed.

    TODO: once ingestion carries `start_seconds`, `end_seconds` and `speaker` on
    each chunk, compute talk-over, dead air and words-per-minute here in code and
    judge them with `AUDIO_METRICS_SYSTEM_PROMPT`.
    """
    item = task["item"]
    finding = build_finding(item, task["retrieved"], [], "N/A", AUDIO_METRICS_UNAVAILABLE)
    return {"findings": [finding], "spans": []}
