"""Graph wiring for the compliance pipeline (CLAUDE.md "Pipeline stages").

START -> ingest -> quality_check -> role_inference -> call_intent
call_intent --Send per Stage 2 item--> check_item   (no items -> verify directly)
check_item -> verify -> decide -> aggregate -> output -> END

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

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from compliance_pipeline.config import active_checklist
from compliance_pipeline.nodes.aggregate import aggregate_node
from compliance_pipeline.nodes.call_intent import call_intent_node
from compliance_pipeline.nodes.check_item import check_item_node
from compliance_pipeline.nodes.decide import decide_node
from compliance_pipeline.nodes.ingest import ingest_node
from compliance_pipeline.nodes.output import output_node
from compliance_pipeline.nodes.quality_check import quality_check_node
from compliance_pipeline.nodes.role_inference import role_inference_node
from compliance_pipeline.nodes.verify import verify_node
from compliance_pipeline.state import CheckItemTask, ComplianceState, InputState, OutputState


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
    if not applicable:
        return "verify"
    segments = state.get("segments", [])
    call_id = state.get("call_id", "unknown")
    call_intent = state.get("call_intent")
    quality_flags = state.get("quality_flags", [])
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


def build_graph():
    builder = StateGraph(ComplianceState, input_schema=InputState, output_schema=OutputState)

    builder.add_node("ingest", ingest_node)
    builder.add_node("quality_check", quality_check_node)
    builder.add_node("role_inference", role_inference_node)
    builder.add_node("call_intent", call_intent_node)
    builder.add_node("check_item", check_item_node)
    builder.add_node("verify", verify_node)
    builder.add_node("decide", decide_node)
    builder.add_node("aggregate", aggregate_node)
    builder.add_node("output", output_node)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "quality_check")
    builder.add_edge("quality_check", "role_inference")
    builder.add_edge("role_inference", "call_intent")
    builder.add_conditional_edges("call_intent", _route_to_checks, ["check_item", "verify"])
    builder.add_edge("check_item", "verify")
    builder.add_edge("verify", "decide")
    builder.add_edge("decide", "aggregate")
    builder.add_edge("aggregate", "output")
    builder.add_edge("output", END)

    return builder.compile()


graph = build_graph()
