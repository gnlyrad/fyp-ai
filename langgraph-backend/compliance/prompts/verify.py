"""Prompt for `nodes/verify.py`.

TODO(step 8): ask whether the cited `evidence_ids` actually support a FAIL,
a red flag, or a result with confidence < 0.7. If not, the caller downgrades
the result to CANNOT_DETERMINE.
"""
from __future__ import annotations

SYSTEM = "TODO: verification system prompt."


def build_user(result: dict, evidence_text: str) -> str:
    """TODO: format one result plus its cited segments for the verify call."""
    raise NotImplementedError
