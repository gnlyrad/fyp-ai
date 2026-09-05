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

    spans = sorted(state.get("spans", []), key=lambda s: s["start"])
    labels = Counter(s["label"] for s in spans)

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
                # Any Fail, or an Unclear rate above a third, warrants a human review.
                # N/A findings — which today includes every audio-metrics item —
                # are not Unclear, so routed-but-unimplemented checks cannot on
                # their own trip the review flag.
                "requires_review": counts.get("Fail", 0) > 0
                or counts.get("Unclear", 0) > len(findings) / 3,
                "by_type": by_type,
            },
            "findings": findings,
            "spans": spans,
        }
    }
