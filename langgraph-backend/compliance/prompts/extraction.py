"""Prompt for `nodes/extraction.py`.

TODO(step 4): extract every order discussed, marking corrections
(`superseded`) and whether each was explicitly confirmed by the client.
`normalise.py` fills in `product_id` afterwards; the LLM never sets it.
"""
from __future__ import annotations

SYSTEM = "TODO: order/entity extraction system prompt."


def build_user(transcript: str) -> str:
    """TODO: format the transcript for the extraction call."""
    raise NotImplementedError
