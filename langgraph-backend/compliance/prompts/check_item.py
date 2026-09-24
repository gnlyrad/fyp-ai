"""Prompt for `nodes/check_item.py`.

TODO(step 7): the prompt sees only `question`, `sub_criteria`, the cached
rule text for this item, the transcript segments, and the context data keys
listed in `needs_data` -- never the whole checklist config. C06 additionally
asks for `RedFlag`s.
"""
from __future__ import annotations

SYSTEM = "TODO: checklist item evaluation system prompt."


def build_user(
    question: str,
    sub_criteria: list[dict],
    rules: list[dict],
    transcript: str,
    context: dict,
) -> str:
    """TODO: assemble the per-item check prompt from config + retrieved rules + context."""
    raise NotImplementedError
