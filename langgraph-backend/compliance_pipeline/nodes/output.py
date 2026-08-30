"""Output node: fold the per-item findings into one auditable report."""
from __future__ import annotations

from collections import Counter
from typing import Any

from ..prompts.checklist import CHECKLIST
from ..state import ComplianceState

# Checklist order is the report order; Send fan-out completes in arbitrary order.
_ORDER = {item["id"]: index for index, item in enumerate(CHECKLIST)}


def output_node(state: ComplianceState) -> dict[str, Any]:
    findings = sorted(state["findings"], key=lambda f: _ORDER.get(f["checklist_id"], 999))
    counts = Counter(f["verdict"] for f in findings)
    call = state.get("call", {})

    return {
        "report": {
            "call_id": state["call_id"],
            "title": call.get("title"),
            "business_area": call.get("business_area"),
            "risk_level": call.get("risk_level"),
            "summary": {
                "total": len(findings),
                "pass": counts.get("Pass", 0),
                "fail": counts.get("Fail", 0),
                "unclear": counts.get("Unclear", 0),
                "not_applicable": counts.get("N/A", 0),
                # Any Fail, or an Unclear rate above a third, warrants a human review.
                "requires_review": counts.get("Fail", 0) > 0
                or counts.get("Unclear", 0) > len(findings) / 3,
            },
            "findings": findings,
        }
    }
