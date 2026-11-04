"""Graph wiring for the compliance pipeline (CLAUDE.md "Pipeline stages").

START -> ingest -> quality_check -> role_inference -> call_intent
call_intent --Send per Stage 2 item--> check_item   (no items -> verify directly)
check_item -> verify -> decide -> aggregate -> output -> END

Optional cache warm-up runs the first ordinary check/group before the rest.
It adds no priming inference and never feeds that answer to sibling checks.

No extraction/normalise stage: the pipeline does not pull structured
entities out of the transcript. Product facts are inferred from the
transcript itself (what the advisor says about the product), judged against
MAS rules from RAG.

`gate` (nodes/gate.py) is implemented but not wired in yet: every Stage 2
item goes to `check_item`. To enable it, add the node between `call_intent`
and the fan-out; `_route_to_checks` already honours `applicable_items`.

No context_lookup stage either: there is no client master or restricted-list
table. Identity, eligibility and red flags are judged from the transcript
against MAS rules from RAG.
"""
from __future__ import annotations

import os

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from compliance_pipeline.config import active_checklist
from compliance_pipeline.nodes.aggregate import aggregate_node
from compliance_pipeline.nodes.call_intent import call_intent_node
from compliance_pipeline.nodes.check_item import check_item_node
from compliance_pipeline.nodes.check_batch import check_batch_node, check_batch_size
from compliance_pipeline.nodes.decide import decide_node
from compliance_pipeline.nodes.ingest import ingest_node
from compliance_pipeline.nodes.output import output_node
from compliance_pipeline.nodes.quality_check import quality_check_node
from compliance_pipeline.nodes.role_inference import role_inference_node
from compliance_pipeline.nodes.verify import verify_node
from compliance_pipeline.state import CheckItemTask, ComplianceState, InputState, OutputState
from compliance_pipeline.usage import track_usage
from compliance_pipeline.observability import begin_observation, finish_observation, track_stage


def _route_to_checks(state: ComplianceState) -> list[Send] | str:
    """One `check_item` branch per item to check, or straight to `verify` if none.

    Uses `applicable_items` when `gate` has set it; otherwise every Stage 2
    item of this run's checklist (the caller's, or the default). Each branch
    gets its full item config, so caller-added items work too.
    """
    checklist = {item.id: item for item in active_checklist(state)}
    applicable = state.get("applicable_items")
    if applicable is None:
        applicable = [item_id for item_id, item in checklist.items() if item.stage == 2]
    warmed = set(state.get("warmed_check_ids") or [])
    applicable = [item_id for item_id in applicable if item_id not in warmed]
    if not applicable:
        return "verify"
    segments = state.get("segments", [])
    call_id = state.get("call_id", "unknown")
    call_intent = state.get("call_intent")
    quality_flags = state.get("quality_flags", [])
    batch_size = check_batch_size()
    if batch_size > 1:
        return [Send("check_batch", {
            "checklist_items": [checklist[item_id].model_dump() for item_id in applicable[start:start + batch_size]],
            "call_id": call_id, "segments": segments, "call_intent": call_intent,
            "quality_flags": quality_flags,
        }) for start in range(0, len(applicable), batch_size)]
    return [
        Send(
            "check_item",
            CheckItemTask(
                item_id=item_id,
                checklist_item=checklist[item_id].model_dump() if item_id in checklist else None,
                call_id=call_id,
                segments=segments,
                call_intent=call_intent,
                quality_flags=quality_flags,
            ),
        )
        for item_id in applicable
    ]


def _route_after_intent(state):
    value = os.getenv("COMPLIANCE_CHECK_CACHE_WARMUP", "0")
    if value not in {"0", "1"}:
        raise ValueError("COMPLIANCE_CHECK_CACHE_WARMUP must be 0 or 1")
    tasks = _route_to_checks(state)
    # Warm only the online path. One ordinary request/group goes first;
    # it is retained, never repeated, and its answer is not fed to siblings.
    if (value == "1" and isinstance(tasks, list) and len(tasks) > 1
            and os.getenv("COMPLIANCE_CHECK_TRANSPORT", "standard") == "standard"):
        return "warm_checks"
    return tasks


def warm_checks_node(state):
    tasks = _route_to_checks(state)
    if not isinstance(tasks, list):
        return {}
    first = tasks[0]
    if first.node == "check_batch":
        update = check_batch_node(first.arg)
        ids = [item["id"] for item in first.arg["checklist_items"]]
    else:
        update = check_item_node(first.arg)
        ids = [first.arg["item_id"]]
    return {**update, "warmed_check_ids": ids}


def build_graph():
    builder = StateGraph(ComplianceState, input_schema=InputState, output_schema=OutputState)

    builder.add_node("observe_start", begin_observation)
    builder.add_node("ingest", track_stage(ingest_node))
    builder.add_node("quality_check", track_stage(quality_check_node))
    builder.add_node("role_inference", track_usage(role_inference_node))
    builder.add_node("call_intent", track_usage(call_intent_node))
    builder.add_node("check_item", track_usage(check_item_node))
    builder.add_node("check_batch", track_usage(check_batch_node))
    builder.add_node("warm_checks", track_usage(warm_checks_node))
    builder.add_node("verify", track_usage(verify_node))
    builder.add_node("decide", track_usage(decide_node))
    builder.add_node("aggregate", track_stage(aggregate_node))
    builder.add_node("observe_finish", finish_observation)
    builder.add_node("output", output_node)

    builder.add_edge(START, "observe_start")
    builder.add_edge("observe_start", "ingest")
    builder.add_edge("ingest", "quality_check")
    builder.add_edge("quality_check", "role_inference")
    builder.add_edge("role_inference", "call_intent")
    builder.add_conditional_edges("call_intent", _route_after_intent,
                                  ["warm_checks", "check_item", "check_batch", "verify"])
    builder.add_conditional_edges("warm_checks", _route_to_checks, ["check_item", "check_batch", "verify"])
    builder.add_edge("check_item", "verify")
    builder.add_edge("check_batch", "verify")
    builder.add_edge("verify", "decide")
    builder.add_edge("decide", "aggregate")
    builder.add_edge("aggregate", "observe_finish")
    builder.add_edge("observe_finish", "output")
    builder.add_edge("output", END)

    return builder.compile()


graph = build_graph()
