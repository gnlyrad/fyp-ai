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
  from there. `final_red_flags` is the same for red flags (verified list when
  `verify` ran, else the raw `red_flags`).
- `summary`, `speaker_roles`, `dual_channel`: bookkeeping the plan
  calls for but CLAUDE.md's `ComplianceState` listing predates.
- `item_results` (CLAUDE.md calls it `results`): the caller's request uses
  `results` for the raw AWS Transcribe object, so the `operator.add`
  `ItemResult` accumulator is named `item_results` to avoid a channel clash.
- No `orders`/`product_context`: the pipeline does not extract structured
  entities. Product facts are inferred from the transcript itself, judged
  against MAS rules from RAG.
- No `client_context`/`restricted_context`/`client_id`: there is no
  context_lookup stage; every check works from the transcript and RAG rules.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from compliance_pipeline.schemas import Action, ItemResult, RedFlag


class CheckItemTask(TypedDict, total=False):
    """The `Send` payload from `gate` to one `check_item` branch."""

    item_id: str
    checklist_item: dict | None  # the item's config (ChecklistItem dump) for this run
    call_id: str
    segments: list[dict]
    call_intent: dict | None
    quality_flags: list[str]


class ComplianceState(TypedDict, total=False):
    call_id: str
    s3_url: str  # S3 location of the Transcribe output JSON; ingest fetches it
    raw_transcript: dict
    results: dict  # AWS Transcribe `results` object (preferred over raw_transcript)
    checklist: list[dict]  # this run's checklist (ChecklistItem dumps), resolved by ingest
    using_default_checklist: bool  # distinguish default IDs from caller-defined IDs
    transcript: str  # formatted transcript, one line per turn
    segments: list[dict]  # {id, speaker, turn, start, end, text, role, confidence, word_count}
    quality_flags: list[str]
    call_intent: dict | None  # {category, confidence, evidence, reached, reasoning}
    applicable_items: list[str]
    item_results: Annotated[list[ItemResult], operator.add]
    red_flags: Annotated[list[RedFlag], operator.add]
    actions: list[Action]
    escalation: dict | None  # {required, reasons, hold_transaction}
    errors: Annotated[list[str], operator.add]

    # Deviations from CLAUDE.md's ComplianceState, see module docstring.
    verified_results: list[ItemResult]
    verified_red_flags: list[RedFlag]
    final_red_flags: list[RedFlag]
    final_results: list[ItemResult]
    summary: dict
    speaker_roles: dict | None
    dual_channel: bool


class InputState(TypedDict, total=False):
    """What callers pass in to invoke the graph.

    `s3_url` (where the AWS Transcribe output JSON lives: `s3://bucket/key`,
    an S3 HTTPS URL, or a presigned URL) is the normal input; `ingest`
    fetches the file itself. It takes precedence over `results` (the AWS
    Transcribe `results` object, as in the caller request's
    `input["results"]`), which takes precedence over `raw_transcript` (a
    full Transcribe job JSON, or its `results` object).
    """

    s3_url: str  # S3 location of the Transcribe output JSON; takes precedence
    raw_transcript: dict  # AWS Transcribe job JSON, or its "results" object
    results: dict  # AWS Transcribe `results` object; takes precedence
    checklist: list[dict]  # optional; replaces the default Stage 2 checks (see config.resolve_checklist)
    call_id: str
    speaker_roles: dict


class OutputState(TypedDict, total=False):
    """What the graph returns."""

    call_id: str
    transcript: str
    segments: list[dict]
    quality_flags: list[str]
    call_intent: dict | None
    checklist: list[dict]  # the checklist this run used
    final_red_flags: list[RedFlag]
    actions: list[Action]
    escalation: dict | None
    summary: dict
    final_results: list[ItemResult]
    errors: list[str]
