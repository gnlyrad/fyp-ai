"""Graph wiring: ingestion -> classification -> retrieval -> (fan-out) analysis -> output.

This is a *workflow*, not an agent loop: the sequence is fixed and the LLM never
chooses the route. Two things are dynamic, and both are decided before any
judgement is made — the fan-out width, which is the length of the checklist, and
which analysis node each item lands on, which is its `question_type`. That is
what keeps the pipeline auditable: every run visits the same stages in the same
order, every checklist item gets exactly one judgement, and the routing decision
is recorded on the item as `type_rationale`.

Classification sits before retrieval because the type decides how much of the
transcript retrieval hands over, not just how it is judged.
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .nodes.analysis import (
    analysis_audio_node,
    analysis_deterministic_node,
    analysis_exhaustive_node,
    analysis_targeted_node,
)
from .nodes.classification import classification_node
from .nodes.ingestion import ingestion_node
from .nodes.output import output_node
from .nodes.retrieval import retrieval_node
from .state import DEFAULT_QUESTION_TYPE, AnalyseInput, ComplianceState

ANALYSIS_NODES = {
    "deterministic": ("analysis_deterministic", analysis_deterministic_node),
    "exhaustive": ("analysis_exhaustive", analysis_exhaustive_node),
    "targeted": ("analysis_targeted", analysis_targeted_node),
    "audio_metrics": ("analysis_audio", analysis_audio_node),
}
NODE_BY_TYPE = {qtype: name for qtype, (name, _) in ANALYSIS_NODES.items()}
DEFAULT_ANALYSIS_NODE = NODE_BY_TYPE[DEFAULT_QUESTION_TYPE]


def fan_out_to_analysis(state: ComplianceState) -> list[Send]:
    """Dispatch each checklist item to the analysis node for its question type.

    Each `Send` carries its own payload, so an analysis node never sees the whole
    state and items cannot leak evidence into one another.
    """
    evidence = state["evidence"]
    transcript = state["transcript"]
    return [
        Send(NODE_BY_TYPE.get(item.get("question_type"), DEFAULT_ANALYSIS_NODE), {
            "item": item,
            "retrieved": evidence.get(item["id"], []),
            "transcript": transcript,
        })
        for item in state["checklist"]
    ]


builder = StateGraph(ComplianceState, input_schema=AnalyseInput)

builder.add_node("ingestion", ingestion_node)
builder.add_node("classification", classification_node)
builder.add_node("retrieval", retrieval_node)
for name, node in ANALYSIS_NODES.values():
    builder.add_node(name, node)
builder.add_node("output", output_node)

builder.add_edge(START, "ingestion")
builder.add_edge("ingestion", "classification")
builder.add_edge("classification", "retrieval")
builder.add_conditional_edges("retrieval", fan_out_to_analysis, list(NODE_BY_TYPE.values()))
for name in NODE_BY_TYPE.values():
    builder.add_edge(name, "output")
builder.add_edge("output", END)

graph = builder.compile()
