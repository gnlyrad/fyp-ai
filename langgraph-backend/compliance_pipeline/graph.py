"""Fixed workflow with applicability, evidence review and consistency checks.

Each checklist item is judged separately. The final consistency stage reviews
cross-call claims before output aggregates findings and review reasons.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .nodes.analysis import (
    analysis_deterministic_node,
    analysis_exhaustive_node,
    analysis_targeted_node,
)
from .nodes.classification import classification_node
from .nodes.applicability import applicability_node
from .nodes.consistency import consistency_node
from .nodes.guidance import guidance_node
from .nodes.ingestion import ingestion_node
from .nodes.output import output_node
from .nodes.retrieval import retrieval_node
from .state import DEFAULT_QUESTION_TYPE, AnalyseInput, ComplianceState

ANALYSIS_NODES = {
    "deterministic": ("analysis_deterministic", analysis_deterministic_node),
    "exhaustive": ("analysis_exhaustive", analysis_exhaustive_node),
    "targeted": ("analysis_targeted", analysis_targeted_node),
}
NODE_BY_TYPE = {qtype: name for qtype, (name, _) in ANALYSIS_NODES.items()}
DEFAULT_ANALYSIS_NODE = NODE_BY_TYPE[DEFAULT_QUESTION_TYPE]


def fan_out_to_analysis(state: ComplianceState) -> list[Send]:
    """Dispatch each checklist item to the analysis node for its question type.

    Each task receives its own requirement, initial retrieval and the source
    chunks needed for an explicit full-call review.
    """
    evidence = state["evidence"]
    guidance = state.get("guidance", {})
    transcript = state["transcript"]
    return [
        Send(NODE_BY_TYPE.get(item.get("question_type"), DEFAULT_ANALYSIS_NODE), {
            "item": item,
            "retrieved": evidence.get(item["id"], []),
            "transcript": transcript,
            "guidance": guidance.get(item["id"], []),
            "chunks": state["chunks"],
            "applicability": state.get("applicability", {}).get(item["id"], {}),
            "retrieval_diagnostics": state.get("retrieval_diagnostics", {}).get(item["id"], {}),
            "context_mode": state["context_mode"],
            "analysis_max_attempts": state["analysis_max_attempts"],
        })
        for item in state["checklist"]
    ]


builder = StateGraph(ComplianceState, input_schema=AnalyseInput)

builder.add_node("ingestion", ingestion_node)
builder.add_node("classification", classification_node)
builder.add_node("applicability", applicability_node)
builder.add_node("consistency", consistency_node)
builder.add_node("guidance", guidance_node)
builder.add_node("retrieval", retrieval_node)
for name, node in ANALYSIS_NODES.values():
    builder.add_node(name, node)
builder.add_node("output", output_node)

builder.add_edge(START, "ingestion")
builder.add_edge("ingestion", "classification")
builder.add_edge("classification", "applicability")
builder.add_edge("applicability", "guidance")
builder.add_edge("guidance", "retrieval")
builder.add_conditional_edges("retrieval", fan_out_to_analysis, list(NODE_BY_TYPE.values()))
for name in NODE_BY_TYPE.values():
    builder.add_edge(name, "consistency")
builder.add_edge("consistency", "output")
builder.add_edge("output", END)

graph = builder.compile()
