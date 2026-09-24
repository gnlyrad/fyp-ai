"""Prompt for `nodes/call_intent.py`.

TODO(step 3): classify the call as enquiry, execution_only, or advisory,
using the most regulated intent reached anywhere in the call (a call can
move from enquiry to order mid-call).
"""
from __future__ import annotations

SYSTEM = "TODO: call intent classification system prompt."


def build_user(transcript: str) -> str:
    """TODO: format the transcript for the call-intent classification call."""
    raise NotImplementedError
