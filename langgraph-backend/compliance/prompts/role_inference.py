"""Prompt for `nodes/role_inference.py`.

TODO(step 2): map each speaker label (spk_0, spk_1, ...) seen in the first
~12 segments to Advisor / Client / uncertain, without inferring the role from
who places an order (see `compliance_pipeline/nodes/ingestion.py`'s
SPEAKER_ROLE_PROMPT for the rule this carries forward).
"""
from __future__ import annotations

SYSTEM = "TODO: role inference system prompt."


def build_user(segments: list[dict]) -> str:
    """TODO: format the first ~12 segments for the role-inference call."""
    raise NotImplementedError
