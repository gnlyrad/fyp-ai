"""Output node: fold the per-item findings into one auditable report."""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..state import ComplianceState


def output_node(state: ComplianceState) -> dict[str, Any]:
    # Report order follows the checklist as submitted; Send fan-out completes in
    # arbitrary order, so the findings have to be put back in the caller's order.
    checklist = state.get("checklist", [])
    order = {item["id"]: index for index, item in enumerate(checklist)}

    findings = sorted(state["findings"], key=lambda f: order.get(f["checklist_id"], 999))
    counts = Counter(f["verdict"] for f in findings)

    spans = sorted(state.get("spans", []), key=lambda s: s["start"])
    labels = Counter(s["label"] for s in spans)

    call = state.get("call", {})

    return {
        "report": {
            "call_id": state.get("call_id"),
            "title": call.get("title"),
            "business_area": call.get("business_area"),
            "risk_level": call.get("risk_level"),
            "summary": {
                "total": len(findings),
                "pass": counts.get("Pass", 0),
                "fail": counts.get("Fail", 0),
                "unclear": counts.get("Unclear", 0),
                "not_applicable": counts.get("N/A", 0),
                "compliant_spans": labels.get("compliant", 0),
                "violation_spans": labels.get("violation", 0),
                # Any Fail, or an Unclear rate above a third, warrants a human review.
                "requires_review": counts.get("Fail", 0) > 0
                or counts.get("Unclear", 0) > len(findings) / 3,
            },
            "findings": findings,
            "spans": spans,
        }
    }
