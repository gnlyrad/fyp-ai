"""`call_intent`: classify the call as enquiry / execution_only / advisory.

One LLM call over the formatted transcript, asking which intents were
*reached* during the call (with evidence quotes), not the final label
directly. Code then picks the most regulated intent reached -- advisory >
execution_only > enquiry -- from whatever observations survive evidence
resolution (CLAUDE.md "Prefer plain code over an LLM whenever the task is
deterministic"). A call can move from enquiry to order mid-call, so this
picks the most regulated intent reached anywhere in the call, not just the
last one. Runs right after `role_inference`, feeding `gate`.

`category` (and the C02 `ItemResult`) is CANNOT_DETERMINE / None when: the
LLM call failed twice, the LLM says this isn't a financial-services call,
no observation survives evidence resolution, or confidence is below
`MIN_INTENT_CONFIDENCE`.

Each `IntentObservation`'s evidence quotes are resolved in code via
`nodes/_common.py:resolve_evidence` into persisted `EvidenceRef`s. An
observation survives only if it resolves to at least one ref (verified or
not); if every ref for a surviving observation is unverified, it is still
kept but an `errors` note is added.
"""
from __future__ import annotations

from compliance_pipeline.config import load_checklist
from compliance_pipeline.llm import get_llm_response
from compliance_pipeline.nodes._common import format_transcript, resolve_evidence
from compliance_pipeline.prompts.call_intent import SYSTEM, build_user
from compliance_pipeline.schemas import CallIntent, CallIntentOutput, EvidenceRef, ItemResult, Status
from compliance_pipeline.state import ComplianceState

MIN_INTENT_CONFIDENCE = 0.6

# Most regulated first. Code, not the LLM, picks the winner from this order.
_INTENT_RANK = ["advisory", "execution_only", "enquiry"]

ITEM_ID = "C02"


def _item_id() -> str:
    """The stage-1 checklist item with `answer_type: category` (C02), read
    from config rather than hard-coded where that's easy to do."""
    for item in load_checklist():
        if item.stage == 1 and item.answer_type == "category":
            return item.id
    return ITEM_ID


def _empty_call_intent() -> dict:
    return {"category": None, "confidence": 0.0, "evidence": [], "reached": [], "reasoning": ""}


def call_intent_node(state: ComplianceState) -> dict:
    item_id = _item_id()
    segments = state.get("segments") or []

    if not segments:
        result = ItemResult(
            item_id=item_id,
            status=Status.CANNOT_DETERMINE,
            confidence=0.0,
            evidence=[],
            reasoning="No transcript segments to classify.",
        )
        return {"call_intent": _empty_call_intent(), "item_results": [result]}

    transcript = format_transcript(segments)

    errors: list[str] = []
    result = get_llm_response("call_intent", SYSTEM, build_user(transcript), CallIntentOutput)

    if not result.ok:
        errors.append(f"call_intent: {result.error}")
        item_result = ItemResult(
            item_id=item_id,
            status=Status.CANNOT_DETERMINE,
            confidence=0.0,
            evidence=[],
            reasoning="Call intent classification failed.",
            execution_status="prediction_failure",
            failure_category="classification_unavailable",
            failure_stage="call_intent",
        )
        return {"call_intent": _empty_call_intent(), "item_results": [item_result], "errors": errors}

    output = result.value

    # Resolve each observation's evidence quotes; drop observations with no
    # surviving ref, note observations whose refs are all unverified.
    surviving_observations: list[tuple[str, list[EvidenceRef]]] = []
    for obs in output.observations:
        refs, notes = resolve_evidence(segments, obs.evidence)
        errors.extend(f"call_intent: {note}" for note in notes)
        if not refs:
            continue
        if all(not ref.verified for ref in refs):
            errors.append(
                f"call_intent: observation {obs.category!r} kept with only unverified evidence"
            )
        surviving_observations.append((obs.category, refs))

    reached_ranked = [
        cat for cat in _INTENT_RANK
        if any(o_cat == cat for o_cat, _ in surviving_observations)
    ]

    category = None
    evidence: list[EvidenceRef] = []
    if (
        not output.not_financial_call
        and surviving_observations
        and output.confidence >= MIN_INTENT_CONFIDENCE
    ):
        category = reached_ranked[0]
        seen: set[tuple[str, int, int, bool]] = set()
        for o_cat, refs in surviving_observations:
            if o_cat != category:
                continue
            for ref in refs:
                key = (ref.segment_id, ref.start_char, ref.end_char, ref.verified)
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(ref)

    call_intent = CallIntent(
        category=category,
        confidence=output.confidence,
        evidence=evidence,
        reached=reached_ranked,
        reasoning=output.reasoning,
    ).model_dump()

    if category is not None:
        status = Status.PASS
        others = [c for c in reached_ranked if c != category]
        suffix = f" (also: {', '.join(others)})" if others else ""
        reasoning = f"Most regulated intent reached: {category}{suffix}. {output.reasoning}".strip()
    else:
        status = Status.CANNOT_DETERMINE
        reasoning = f"Could not determine call intent. {output.reasoning}".strip()

    item_result = ItemResult(
        item_id=item_id,
        status=status,
        confidence=output.confidence,
        evidence=evidence,
        reasoning=reasoning,
    )

    out = {"call_intent": call_intent, "item_results": [item_result]}
    if errors:
        out["errors"] = errors
    return out
