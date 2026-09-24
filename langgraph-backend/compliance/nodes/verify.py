"""`verify`: re-check every FAIL, red flag, and low-confidence result.

Final intended behaviour (CLAUDE.md): in code, confirm every `evidence_id`
still exists in `segments`. For every FAIL, every red flag, and every result
with confidence < 0.7, ask the LLM whether the cited segments actually
support the result; if not, downgrade to CANNOT_DETERMINE and note why.
Because `results`/`red_flags` are `operator.add` reducers, replacements are
written to the plain-overwrite `verified_results`/`verified_red_flags` keys
instead (see `state.py` docstring) -- `aggregate` prefers these over
`item_results`/`red_flags` for the same id.

Step-0 stub: no verification performed yet.
"""
from __future__ import annotations

from compliance.state import ComplianceState


def verify_node(state: ComplianceState) -> dict:
    return {"verified_results": [], "verified_red_flags": []}
