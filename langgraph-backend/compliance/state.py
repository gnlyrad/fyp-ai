"""Graph state (CLAUDE.md "State" section), plus the input/output schemas.

Deviations from CLAUDE.md, documented per the plan:
- `verified_results` / `verified_red_flags`: `item_results` and `red_flags` are
  `operator.add` reducers (parallel `check_item` branches append to them), so
  `verify` cannot edit them in place. It writes replacements to these two
  plain-overwrite keys instead; `aggregate` prefers a verified entry over the
  matching `item_results`/`red_flags` entry for the same id.
- `final_results`: `aggregate` cannot write back to `item_results` (it is an
  add-reducer, so a write would append rather than replace). It writes the
  deduplicated, gap-filled list to `final_results` instead, and `output` reads
  from there.
- `summary`, `client_id`, `speaker_roles`, `dual_channel`: bookkeeping the plan
  calls for but CLAUDE.md's `ComplianceState` listing predates.
- `item_results` (CLAUDE.md calls it `results`): the caller's request uses
  `results` for the raw AWS Transcribe object, so the `operator.add`
  `ItemResult` accumulator is named `item_results` to avoid a channel clash.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from compliance.schemas import Action, ItemResult, Order, RedFlag


class CheckItemTask(TypedDict, total=False):
    """The `Send` payload from `gate` to one `check_item` branch."""

    item_id: str
    call_id: str
    segments: list[dict]
    client_context: dict | None
    product_context: list[dict]
    orders: list[dict]
    call_intent: dict | None


class ComplianceState(TypedDict, total=False):
    call_id: str
    raw_transcript: dict
    results: dict  # AWS Transcribe `results` object (preferred over raw_transcript)
    checklist: list[dict]  # caller-supplied checklist items, carried through untouched
    transcript: str  # formatted transcript, one line per turn
    segments: list[dict]  # {id, speaker, turn, start, end, text, role, confidence, word_count}
    quality_flags: list[str]
    call_intent: dict | None  # {category, confidence, evidence_ids}
    orders: list[Order]
    client_context: dict | None
    product_context: list[dict]
    applicable_items: list[str]
    item_results: Annotated[list[ItemResult], operator.add]
    red_flags: Annotated[list[RedFlag], operator.add]
    actions: list[Action]
    escalation: dict | None  # {required, reasons, hold_transaction}
    errors: Annotated[list[str], operator.add]

    # Deviations from CLAUDE.md's ComplianceState, see module docstring.
    verified_results: list[ItemResult]
    verified_red_flags: list[RedFlag]
    final_results: list[ItemResult]
    summary: dict
    client_id: str | None
    speaker_roles: dict | None
    dual_channel: bool


class InputState(TypedDict, total=False):
    """What callers pass in to invoke the graph.

    `results` (the AWS Transcribe `results` object, as in the caller
    request's `input["results"]`) takes precedence
    over `raw_transcript` (a full Transcribe job JSON, or its `results`
    object) when both are given.
    """

    raw_transcript: dict  # AWS Transcribe job JSON, or its "results" object
    results: dict  # AWS Transcribe `results` object; takes precedence
    checklist: list[dict]  # caller checklist, carried through untouched
    call_id: str
    client_id: str
    speaker_roles: dict


class OutputState(TypedDict, total=False):
    """What the graph returns."""

    call_id: str
    transcript: str
    segments: list[dict]
    quality_flags: list[str]
    call_intent: dict | None
    orders: list[Order]
    red_flags: list[RedFlag]
    actions: list[Action]
    escalation: dict | None
    summary: dict
    final_results: list[ItemResult]
    errors: list[str]
