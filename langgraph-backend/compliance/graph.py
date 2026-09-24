"""Graph wiring for the compliance pipeline (CLAUDE.md "Pipeline stages").

START -> ingest -> quality_check -> role_inference -> {call_intent, extraction}
extraction -> normalise; [call_intent, normalise] -> context_lookup -> gate
gate --Send per applicable item--> check_item   (no applicable items -> verify directly)
check_item -> verify -> decide -> aggregate -> output -> END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from compliance.nodes.aggregate import aggregate_node
from compliance.nodes.call_intent import call_intent_node
from compliance.nodes.check_item import check_item_node
from compliance.nodes.context_lookup import context_lookup_node
from compliance.nodes.decide import decide_node
from compliance.nodes.extraction import extraction_node
from compliance.nodes.gate import gate_node
from compliance.nodes.ingest import ingest_node
from compliance.nodes.normalise import normalise_node
from compliance.nodes.output import output_node
from compliance.nodes.quality_check import quality_check_node
from compliance.nodes.role_inference import role_inference_node
from compliance.nodes.verify import verify_node
from compliance.state import CheckItemTask, ComplianceState, InputState, OutputState


def _route_from_gate(state: ComplianceState) -> list[Send] | str:
    """One `check_item` branch per applicable item, or straight to `verify` if none."""
    applicable = state.get("applicable_items", [])
    if not applicable:
        return "verify"
    segments = state.get("segments", [])
    call_id = state.get("call_id", "unknown")
    client_context = state.get("client_context")
    product_context = state.get("product_context", [])
    orders = state.get("orders", [])
    call_intent = state.get("call_intent")
    return [
        Send(
            "check_item",
            CheckItemTask(
                item_id=item_id,
                call_id=call_id,
                segments=segments,
                client_context=client_context,
                product_context=product_context,
                orders=orders,
                call_intent=call_intent,
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
    builder.add_node("extraction", extraction_node)
    builder.add_node("normalise", normalise_node)
    builder.add_node("context_lookup", context_lookup_node)
    builder.add_node("gate", gate_node)
    builder.add_node("check_item", check_item_node)
    builder.add_node("verify", verify_node)
    builder.add_node("decide", decide_node)
    builder.add_node("aggregate", aggregate_node)
    builder.add_node("output", output_node)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "quality_check")
    builder.add_edge("quality_check", "role_inference")
    builder.add_edge("role_inference", "call_intent")
    builder.add_edge("role_inference", "extraction")
    builder.add_edge("extraction", "normalise")
    builder.add_edge(["call_intent", "normalise"], "context_lookup")
    builder.add_edge("context_lookup", "gate")
    builder.add_conditional_edges("gate", _route_from_gate, ["check_item", "verify"])
    builder.add_edge("check_item", "verify")
    builder.add_edge("verify", "decide")
    builder.add_edge("decide", "aggregate")
    builder.add_edge("aggregate", "output")
    builder.add_edge("output", END)

    return builder.compile()


graph = build_graph()
