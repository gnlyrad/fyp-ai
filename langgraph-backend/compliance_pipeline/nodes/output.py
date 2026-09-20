"""Output node: fold the per-item findings into one auditable report."""
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


def output_node(state: ComplianceState) -> dict[str, Any]:
    # Report order follows the checklist as submitted; Send fan-out completes in
    # arbitrary order, so the findings have to be put back in the caller's order.
    checklist = state.get("checklist", [])
    order = {item["id"]: index for index, item in enumerate(checklist)}

    findings = sorted(state["findings"], key=lambda f: order.get(f["checklist_id"], 999))
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
    consistency = state.get("consistency", {"status": "not_checked", "conflicts": []})
    review_reasons = []
    if any(f.get("requires_review", f["verdict"] in {"Fail", "Unclear"}) for f in findings):
        review_reasons.append("One or more findings need review.")
    if consistency["status"] not in {"checked", "not_applicable"}:
        review_reasons.append("Cross-call consistency is unresolved or conflicting.")
    # Attribution gaps block only findings that need attribution. Input notices
    # (including timing limitations) remain visible without overriding N/A.

    # Which notices this run was judged against, at a glance. The per-passage
    # citations stay on each finding; this is the cover sheet.
    sources = sorted({
        passage["source"]
        for finding in findings
        for passage in finding.get("regulatory_basis", [])
        if passage.get("source")
    })

    return {
        "report": {
            "summary": {
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
            },
            "findings": findings,
            "warnings": list(dict.fromkeys(state.get("input_warnings", []))),
            "spans": spans,
            "consistency": consistency,
            "applicability": state.get("applicability", {}),
            "retrieval_diagnostics": state.get("retrieval_diagnostics", {}),
        }
    }
