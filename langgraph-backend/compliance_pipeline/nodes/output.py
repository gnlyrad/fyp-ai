"""`output`: persist the final call review (CLAUDE.md "output").

Behind a `ResultWriter` protocol so a Postgres writer can be swapped in
later without touching the graph. Only `JsonFileWriter` is functional;
`PostgresWriter` is a visible stub that raises `NotImplementedError` when
selected -- the node catches that (like any writer failure) and reports it
in `errors` instead of crashing the graph.

The JSON payload stores rule chunks once, referenced by id (CLAUDE.md
"Retrieval": "store rule chunks once and reference them by ID"): every
`rule_id` cited in any final result's `rule_refs` is looked up once via
`retrieval.get_rule` and collected into a `rules: {rule_id: {...}}` map;
an id not found in the cache is dropped from the map and noted in `errors`.

`render_report` renders the payload as a plain-text report for the CLI
(`main.py`); it never raises either.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from compliance_pipeline import config, llm, retrieval
from compliance_pipeline.schemas import Status
from compliance_pipeline.state import ComplianceState

SCHEMA_VERSION = "1"


class ResultWriter(Protocol):
    def write(self, payload: dict) -> None: ...


class JsonFileWriter:
    """Writes `<out_dir>/<call_id>.json` with the payload."""

    def __init__(self, out_dir: str = "output") -> None:
        self.out_dir = Path(out_dir)

    def write(self, payload: dict) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        call_id = payload.get("call_id", "unknown")
        path = self.out_dir / f"{call_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


class PostgresWriter:
    """Stub: makes the Postgres path visible in the interface without
    implementing it yet. `output_node` catches this like any other writer
    failure and reports it in `errors` instead of crashing the graph."""

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn

    def write(self, payload: dict) -> None:
        raise NotImplementedError("Postgres writer not implemented yet")


def _dump(items: list) -> list:
    return [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in items]


def _get_writer() -> ResultWriter:
    kind = os.getenv("COMPLIANCE_WRITER", "json")
    out_dir = os.getenv("COMPLIANCE_OUTPUT_DIR", "output")
    if kind == "postgres":
        return PostgresWriter(dsn=os.getenv("COMPLIANCE_POSTGRES_DSN"))
    return JsonFileWriter(out_dir=out_dir)


def _collect_rules(final_results: list, errors: list[str]) -> dict[str, dict]:
    """Every `rule_id` referenced by any result's `rule_refs`, looked up
    once via `retrieval.get_rule` and stored `{rule_id: {text, source,
    page}}`. An id not found in the rule cache is omitted and noted."""
    rules: dict[str, dict] = {}
    seen: set[str] = set()
    for r in final_results:
        rule_refs = getattr(r, "rule_refs", None)
        if rule_refs is None and isinstance(r, dict):
            rule_refs = r.get("rule_refs", [])
        for rule_id in rule_refs or []:
            if rule_id in seen:
                continue
            seen.add(rule_id)
            chunk = retrieval.get_rule(rule_id)
            if chunk is None:
                errors.append(f"output: rule_id {rule_id!r} not found in rule cache; omitted")
                continue
            rules[rule_id] = {
                "text": chunk.get("text", ""),
                "source": chunk.get("source", ""),
                "page": chunk.get("page"),
            }
    return rules


def _dump_item(item) -> dict:
    """A checklist item as it ran, for the payload (dashboard labels)."""
    return item.model_dump(mode="json")


def build_payload(state: ComplianceState) -> dict:
    """Build the JSON payload from graph state. Pure (no I/O); shared by
    `output_node` and tests so payload shape is checked without a writer."""
    errors: list[str] = list(state.get("errors", []) or [])
    final_results = state.get("final_results", [])

    model_id = os.getenv("COMPLIANCE_MODEL_ID", llm.DEFAULT_MODEL_ID)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "call_id": state.get("call_id", "unknown"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": model_id,
            "structured_method": llm.STRUCTURED_METHOD,
            "seed": llm.SEED,
        },
        "transcript": state.get("transcript", ""),
        "segments": state.get("segments", []),
        "quality_flags": state.get("quality_flags", []),
        "call_intent": state.get("call_intent"),
        "checklist": [_dump_item(item) for item in config.active_checklist(state)],
        "results": _dump(final_results),
        "red_flags": _dump(state.get("final_red_flags", [])),
        "actions": _dump(state.get("actions", [])),
        "escalation": state.get("escalation"),
        "summary": state.get("summary", {}),
        "rules": _collect_rules(final_results, errors),
        "errors": errors,
        "usage": state.get("usage", {}),
    }
    return payload


# ---------------------------------------------------------------------------
# Human-readable report, used by `main.py`.
# ---------------------------------------------------------------------------

def render_report(payload: dict, *, quiet: bool = False) -> str:
    """Render `payload` (as returned by `build_payload`) as a readable
    plain-text report. Never raises -- a malformed payload falls back to a
    short note rather than crashing the CLI."""
    try:
        return _render_report(payload, quiet=quiet)
    except Exception as exc:  # noqa: BLE001 - CLI report must never crash
        return f"(failed to render report: {type(exc).__name__}: {exc})"


def _render_report(payload: dict, *, quiet: bool) -> str:
    lines: list[str] = []
    call_id = payload.get("call_id", "unknown")
    lines.append(f"Call: {call_id}")

    quality_flags = payload.get("quality_flags") or []
    lines.append(f"Quality flags: {', '.join(quality_flags) if quality_flags else 'none'}")

    call_intent = payload.get("call_intent") or {}
    lines.append(f"Call intent: {call_intent.get('category') or 'undetermined'}")

    summary = payload.get("summary") or {}
    lines.append("")
    lines.append(
        f"Summary: total={summary.get('total', 0)} counts={summary.get('counts', {})} "
        f"needs_review={summary.get('needs_review')}"
    )
    if summary.get("review_reasons"):
        lines.append("Review reasons:")
        for reason in summary["review_reasons"]:
            lines.append(f"  - {reason}")

    if quiet:
        return "\n".join(lines)

    results = payload.get("results", [])
    segments = payload.get("segments", [])
    lines.append("")
    lines.append("Items:")
    lines.append(f"{'id':<5} {'name':<32} {'status':<16} {'conf':<5} {'ver':<4} warnings")
    names = {c.get("id"): c.get("name") for c in payload.get("checklist") or []}
    for r in results:
        item_id = r.get("item_id", "?")
        cfg_item = config.get_item(item_id)
        name = names.get(item_id) or (cfg_item.name if cfg_item is not None else "?")
        status = r.get("status", "?")
        confidence = r.get("confidence", 0.0)
        verified = r.get("verified", False)
        warnings = r.get("warnings") or []
        lines.append(
            f"{item_id:<5} {name:<32} {status:<16} {confidence:<5.2f} "
            f"{'yes' if verified else 'no':<4} {len(warnings)}"
        )

    escalation = payload.get("escalation") or {}
    lines.append("")
    lines.append(
        f"Escalation: required={escalation.get('required')} "
        f"hold={escalation.get('hold_transaction')} source={escalation.get('source')} "
        f"reasons={escalation.get('reasons') or []}"
    )

    actions = payload.get("actions", [])
    lines.append("")
    lines.append(f"Actions ({len(actions)}):")
    for a in actions:
        lines.append(f"  - [{a.get('owner_role') or '?'}] {a.get('action_type')}: {a.get('description')}")

    red_flags = payload.get("red_flags", [])
    lines.append("")
    lines.append(f"Red flags ({len(red_flags)}):")
    for f in red_flags:
        lines.append(f"  - [{f.get('severity')}/{f.get('category')}] {f.get('description')}")

    lines.append("")
    lines.append("Warnings:")
    warning_count = 0
    for r in results:
        for w in r.get("warnings") or []:
            warning_count += 1
            lines.append(
                f"  - {r.get('item_id')}/{w.get('sub_criterion_id') or '-'} "
                f"[{w.get('severity')}] {w.get('message')}"
            )
    if not warning_count:
        lines.append("  none")

    lines.append("")
    lines.append("FAIL / PARTIAL sub-criteria:")
    any_bad = False
    for r in results:
        if r.get("status") not in (Status.FAIL.value, Status.PARTIAL.value):
            continue
        for sc in r.get("sub_criteria") or []:
            if sc.get("status") != Status.FAIL.value:
                continue
            any_bad = True
            lines.append(f"  - {r.get('item_id')}/{sc.get('id')}: {sc.get('note')}")
            for ev in sc.get("evidence") or []:
                text = _evidence_text_from_segments(segments, ev)
                if len(text) > 120:
                    text = text[:117] + "..."
                lines.append(f"      [{ev.get('segment_id')}] {text!r}")
    if not any_bad:
        lines.append("  none")

    errors = payload.get("errors") or []
    lines.append("")
    lines.append(f"Errors: {len(errors)}")
    for e in errors:
        lines.append(f"  - {e}")

    return "\n".join(lines)


def _evidence_text_from_segments(segments: list[dict], ev: dict) -> str:
    for seg in segments:
        if seg.get("id") == ev.get("segment_id"):
            text = seg.get("text", "") or ""
            return text[ev.get("start_char", 0):ev.get("end_char", 0)]
    return ""


# ---------------------------------------------------------------------------
# Node.
# ---------------------------------------------------------------------------

def output_node(state: ComplianceState) -> dict:
    try:
        payload = build_payload(state)
    except Exception as exc:  # noqa: BLE001 - never crash the graph
        return {"errors": [f"output: failed to build payload: {type(exc).__name__}: {exc}"]}

    try:
        writer = _get_writer()
        writer.write(payload)
    except Exception as exc:  # noqa: BLE001 - never crash the graph
        return {"errors": [f"output: writer failed: {type(exc).__name__}: {exc}"]}

    return {}
