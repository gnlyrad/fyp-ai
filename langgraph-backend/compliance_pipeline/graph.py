"""Graph wiring: ingestion -> retrieval -> (fan-out) analysis -> output.

This is a *workflow*, not an agent loop: the sequence is fixed and the LLM never
chooses the route. The only dynamic element is the fan-out width, which is
determined by the length of the checklist, not by model output. That is what
makes the pipeline auditable — every run visits the same stages in the same
order, and every checklist item gets exactly one judgement.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .nodes.analysis import analysis_node
from .nodes.ingestion import ingestion_node
from .nodes.output import output_node
from .nodes.retrieval import retrieval_node
from .state import AnalyseInput, ComplianceState


def fan_out_to_analysis(state: ComplianceState) -> list[Send]:
    """Dispatch one `analysis_node` invocation per checklist item.

    Each `Send` carries its own payload, so the analysis node never sees the
    whole state and items cannot leak evidence into one another.
    """
    evidence = state["evidence"]
    transcript = state["transcript"]
    return [
        Send("analysis", {
            "item": item,
            "retrieved": evidence.get(item["id"], []),
            "transcript": transcript,
        })
        for item in state["checklist"]
    ]


builder = StateGraph(ComplianceState, input_schema=AnalyseInput)

builder.add_node("ingestion", ingestion_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("analysis", analysis_node)
builder.add_node("output", output_node)

builder.add_edge(START, "ingestion")
builder.add_edge("ingestion", "retrieval")
builder.add_conditional_edges("retrieval", fan_out_to_analysis, ["analysis"])
builder.add_edge("analysis", "output")
builder.add_edge("output", END)

graph = builder.compile()
