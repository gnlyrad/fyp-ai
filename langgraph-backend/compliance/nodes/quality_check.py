"""`quality_check`: flag calls that need extra caution downstream.

Final intended behaviour (CLAUDE.md): flag calls under a set length, average
word confidence below a threshold, and single-speaker calls; also note
dual-channel as informational. Flags do not stop the graph but are shown on
the dashboard and lower confidence in later checks.

Step-0 stub: always returns no flags.
"""
from __future__ import annotations

from compliance.state import ComplianceState

SHORT_CALL_SECONDS = 60
LOW_CONFIDENCE_THRESHOLD = 0.80


def quality_check_node(state: ComplianceState) -> dict:
    return {"quality_flags": []}
