"""Output node: project the internal state into the lean public response shape.

`output_node` no longer builds an internal `report`; it returns the five keys
that make up `AnalyseOutput` (`transcript` passes through from ingestion
unchanged, so this node does not write it): `chunks`, `checklist`, `findings`,
`summary`. Shared data (chunk/question metadata, spans) is stated once and
findings reference chunks by id instead of re-embedding them.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..state import QUESTION_TYPES, ComplianceState, Finding


def verdict_counts(findings: list[Finding]) -> dict[str, int]:
    counts = Counter(f["verdict"] for f in findings)
    return {
        "total": len(findings),
        "pass": counts.get("Pass", 0),
        "fail": counts.get("Fail", 0),
        "unclear": counts.get("Unclear", 0),
        "not_applicable": counts.get("N/A", 0),
    }


def _project_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    projected = {
        "chunk_id": chunk.get("chunk_id"),
        "text": chunk.get("text"),
        "start_seconds": chunk.get("start_seconds"),
        "end_seconds": chunk.get("end_seconds"),
    }
    if "speaker_label" in chunk:
        projected["speaker"] = chunk["speaker_label"]
    return projected


def _project_checklist_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "question_type": item.get("question_type"),
        "pass_condition": item.get("pass_condition"),
    }


def _project_span(span: dict[str, Any]) -> dict[str, Any]:
    return {
        "text": span.get("text"),
        "start": span.get("start"),
        "end": span.get("end"),
        "label": span.get("label"),
        "chunk_id": span.get("chunk_id"),
        "start_seconds": span.get("start_seconds"),
        "end_seconds": span.get("end_seconds"),
    }


def _project_regulatory_basis(passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"source": p.get("source"), "page": p.get("page")} for p in passages]


def _project_finding(finding: Finding) -> dict[str, Any]:
    return {
        "checklist_id": finding.get("checklist_id"),
        "verdict": finding.get("verdict"),
        "reason": finding.get("reason"),
        "evidence_chunk_ids": [c["chunk_id"] for c in finding.get("retrieved_evidence", [])],
        "evidence_status": finding.get("evidence_status"),
        "requires_review": finding.get("requires_review"),
        "regulatory_basis": _project_regulatory_basis(finding.get("regulatory_basis") or []),
        "spans": [_project_span(s) for s in finding.get("spans", [])],
    }


def output_node(state: ComplianceState) -> dict[str, Any]:
    # Output order follows the checklist as submitted; Send fan-out completes in
    # arbitrary order, so the findings have to be put back in the caller's order.
    checklist = state.get("checklist", [])
    order = {item["id"]: index for index, item in enumerate(checklist)}

    findings = sorted(state["raw_findings"], key=lambda f: order.get(f["checklist_id"], 999))
    counts = Counter(f["verdict"] for f in findings)
    # Per-type breakdown: a run that is all-Pass on presence checks but Unclear on
    # every judgement question is a different situation from one that is evenly
    # good, and the headline counts hide that.
    by_type = {
        qtype: verdict_counts([f for f in findings if f.get("question_type") == qtype])
        for qtype in QUESTION_TYPES
    }

    # Same quote under different questions remains distinct; duplicate entries
    # within a finding do not inflate highlights or summary counts.
    unique = {(s["checklist_id"], s["chunk_id"], s["start"], s["end"]): s
              for s in state.get("spans", [])}
    spans = sorted(unique.values(), key=lambda s: s["start"])
    labels = Counter(s["label"] for s in spans)
    review_reasons = []
    if any(f.get("requires_review", f["verdict"] in {"Fail", "Unclear"}) for f in findings):
        review_reasons.append("One or more findings need review.")
    # Only surface the stale consistency reason when consistency_node actually
    # ran; when it is disabled, `consistency` is absent from state, and a
    # default of "not_checked" would otherwise fire this reason on every run.
    consistency = state.get("consistency")
    if consistency is not None and consistency["status"] not in {"checked", "not_applicable"}:
        review_reasons.append("Cross-call consistency is unresolved or conflicting.")

    # Which notices this run was judged against, at a glance. The per-passage
    # citations stay on each finding; this is the cover sheet.
    sources = sorted({
        passage["source"]
        for finding in findings
        for passage in finding.get("regulatory_basis", [])
        if passage.get("source")
    })

    summary = {
        "total": len(findings),
        "pass": counts.get("Pass", 0),
        "fail": counts.get("Fail", 0),
        "unclear": counts.get("Unclear", 0),
        "not_applicable": counts.get("N/A", 0),
        "compliant_spans": labels.get("compliant", 0),
        "violation_spans": labels.get("violation", 0),
        "context_spans": labels.get("context", 0),
        # Any unresolved finding, attribution, coverage or contradiction
        # needs review, even when every other item passed.
        "requires_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "evidence_status_counts": dict(Counter(f.get("evidence_status", "unknown") for f in findings)),
        "by_type": by_type,
        "regulatory_sources": sources,
        "warnings": list(dict.fromkeys(state.get("input_warnings", []))),
    }

    return {
        "chunks": [_project_chunk(c) for c in state.get("chunks", [])],
        "checklist": [_project_checklist_item(i) for i in checklist],
        "findings": [_project_finding(f) for f in findings],
        "summary": summary,
    }
