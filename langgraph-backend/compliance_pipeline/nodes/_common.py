"""Small helpers shared by more than one node."""
from __future__ import annotations

import re

from compliance_pipeline.schemas import (
    EvidenceQuote,
    EvidenceRef,
    ItemResult,
    ReviewWarning,
    Status,
    SubCriterionResult,
)


def format_transcript(segments: list[dict], with_timestamps: bool = False) -> str:
    """Render segments as `[seg_0001] Advisor: text` lines for LLM prompts.

    Falls back to the raw speaker label (e.g. `spk_0`) when `role` has not
    been assigned yet. With `with_timestamps=True`, renders
    `[seg_0001] spk_0 (0.48-16.48): text` instead (used for the human/LLM
    facing `transcript` string built by `ingest`).
    """
    lines = []
    for seg in segments:
        speaker = seg.get("role") or seg.get("speaker") or "unknown"
        if with_timestamps:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            lines.append(f"[{seg['id']}] {speaker} ({start:.2f}-{end:.2f}): {seg.get('text', '')}")
        else:
            lines.append(f"[{seg['id']}] {speaker}: {seg.get('text', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evidence resolution: LLM-cited quotes -> persisted (segment_id, char span) refs.
# ---------------------------------------------------------------------------

_NORMALISE_RE = re.compile(r"[^\w]+", re.UNICODE)


def _normalise(text: str) -> str:
    """Lowercase and collapse everything that isn't a word character, so
    quotes can be matched case/whitespace/punctuation-insensitively."""
    return _NORMALISE_RE.sub(" ", text.lower()).strip()


def _find_normalised(haystack: str, needle: str) -> tuple[int, int] | None:
    """Find `needle` in `haystack` ignoring case, whitespace, and
    punctuation differences, returning the original `(start, end)` char
    offsets into `haystack`, or None if no match.

    Builds a normalised version of `haystack` alongside a mapping from each
    normalised char back to its original index, so the match found in
    normalised space can be translated back to original offsets.
    """
    norm_chars: list[str] = []
    index_map: list[int] = []  # index_map[i] = original index of norm_chars[i]
    prev_was_space = True  # collapse leading/runs of separators like _NORMALISE_RE does
    for i, ch in enumerate(haystack):
        if re.match(r"\w", ch, re.UNICODE):
            norm_chars.append(ch.lower())
            index_map.append(i)
            prev_was_space = False
        else:
            if not prev_was_space and norm_chars:
                norm_chars.append(" ")
                index_map.append(i)
            prev_was_space = True
    # Trim trailing space to mirror `.strip()` in _normalise.
    while norm_chars and norm_chars[-1] == " ":
        norm_chars.pop()
        index_map.pop()

    norm_haystack = "".join(norm_chars)
    norm_needle = _normalise(needle)
    if not norm_needle:
        return None

    pos = norm_haystack.find(norm_needle)
    if pos == -1:
        return None

    start_orig = index_map[pos]
    end_norm = pos + len(norm_needle) - 1
    end_orig = index_map[end_norm] + 1
    return start_orig, end_orig


def resolve_evidence(
    segments: list[dict], quotes: list[EvidenceQuote]
) -> tuple[list[EvidenceRef], list[str]]:
    """Resolve LLM-cited quotes into persisted `EvidenceRef`s.

    - Unknown `segment_id` -> dropped, with a note.
    - Empty quote -> whole segment, `verified=False`.
    - Exact substring match in the segment's `text` -> that span, `verified=True`.
    - Else, case/whitespace/punctuation-insensitive match -> that span (mapped
      back to original offsets), `verified=True`.
    - Else -> whole-segment span `(0, len(text))`, `verified=False`, with a note.

    Identical resulting refs are deduped, preserving first-seen order.
    """
    by_id = {seg["id"]: seg for seg in segments}
    refs: list[EvidenceRef] = []
    seen: set[tuple[str, int, int, bool]] = set()
    notes: list[str] = []

    for q in quotes:
        seg = by_id.get(q.segment_id)
        if seg is None:
            notes.append(f"resolve_evidence: unknown segment_id {q.segment_id!r}; dropped")
            continue

        text = seg.get("text", "") or ""
        quote = q.quote or ""

        if not quote.strip():
            ref = EvidenceRef(segment_id=q.segment_id, start_char=0, end_char=len(text), verified=False)
        else:
            idx = text.find(quote)
            if idx != -1:
                ref = EvidenceRef(
                    segment_id=q.segment_id, start_char=idx, end_char=idx + len(quote), verified=True,
                )
            else:
                match = _find_normalised(text, quote)
                if match is not None:
                    start, end = match
                    ref = EvidenceRef(
                        segment_id=q.segment_id, start_char=start, end_char=end, verified=True,
                    )
                else:
                    ref = EvidenceRef(
                        segment_id=q.segment_id, start_char=0, end_char=len(text), verified=False,
                    )
                    notes.append(
                        f"resolve_evidence: quote not found in {q.segment_id!r}; "
                        "used whole segment, unverified"
                    )

        key = (ref.segment_id, ref.start_char, ref.end_char, ref.verified)
        if key in seen:
            continue
        seen.add(key)
        refs.append(ref)

    return refs, notes


def merged_results(state) -> dict[str, ItemResult]:
    """`item_results` merged with `verified_results`, verified winning per
    `item_id` (mirrors `nodes/aggregate.py`'s merge, without writing to any
    reducer-owned state key)."""
    by_id: dict[str, ItemResult] = {r.item_id: r for r in state.get("item_results", [])}
    for v in state.get("verified_results") or []:
        by_id[v.item_id] = v
    return by_id


def rollup_status(sub_results: list[SubCriterionResult]) -> Status:
    """Derive an item's overall status from its sub-criteria (never trust
    the LLM's own rollup -- CLAUDE.md "Prefer plain code over an LLM
    whenever the task is deterministic"). Used by both `check_item` and
    `verify`, so a downgrade during verification recomputes the same way
    the original result was computed.

    - No non-NOT_APPLICABLE sub-criteria -> NOT_APPLICABLE.
    - Any FAIL -> PARTIAL if something else PASSed, else FAIL.
    - No FAIL but any CANNOT_DETERMINE -> CANNOT_DETERMINE.
    - Otherwise -> PASS.
    """
    non_na = [r for r in sub_results if r.status != Status.NOT_APPLICABLE]
    if not non_na:
        return Status.NOT_APPLICABLE
    if any(r.status == Status.FAIL for r in non_na):
        return Status.PARTIAL if any(r.status == Status.PASS for r in non_na) else Status.FAIL
    if any(r.status == Status.CANNOT_DETERMINE for r in non_na):
        return Status.CANNOT_DETERMINE
    return Status.PASS


def evidence_text(segments: list[dict], ref: EvidenceRef) -> str:
    """The text an `EvidenceRef` points to, sliced from the matching
    segment's `text`. Empty string if the segment is not found."""
    for seg in segments:
        if seg.get("id") == ref.segment_id:
            text = seg.get("text", "") or ""
            return text[ref.start_char:ref.end_char]
    return ""


# Leading junk an LLM sometimes leaks into a free-text field: stray
# punctuation, or its own field syntax ("status: PASS when ...", "note: ...").
_NOTE_PREFIX_RE = re.compile(
    r"^[\s,;:.\-]*(?:(?:status|note|id)\s*:\s*(?:PASS|FAIL|NOT_APPLICABLE|CANNOT_DETERMINE)?\s*"
    r"(?:when\b)?[\s,;:.\-]*)*",
    re.IGNORECASE,
)


def clean_note(note: str | None) -> str:
    """Strip leaked field syntax and stray punctuation from an LLM note."""
    text = _NOTE_PREFIX_RE.sub("", (note or "").strip())
    text = " ".join(text.split())
    return text[:1].upper() + text[1:] if text else ""


# ---------------------------------------------------------------------------
# Reviewer warnings: rule-based, never change status or the rollup.
# ---------------------------------------------------------------------------

def apply_rule_warnings(item: ItemResult, cfg_item) -> None:
    """Add a `source="rule"` warning for every sub-criterion configured with
    `warn_if_unconfirmed: true` whose *current* status is NOT_APPLICABLE or
    CANNOT_DETERMINE (CLAUDE.md "Warnings").

    Idempotent and safe to call more than once on the same item (e.g. once
    in `check_item`, again in `verify` after a downgrade): a sub-criterion
    that already has a warning (of either source) is left alone, except that
    an existing `source="llm"` warning with severity "low" is bumped to at
    least "medium" instead of a duplicate rule warning being added.
    """
    if cfg_item is None:
        return
    flagged = {sc.id: sc.text for sc in cfg_item.sub_criteria if sc.warn_if_unconfirmed}
    if not flagged:
        return

    existing_by_sub: dict[str, list[ReviewWarning]] = {}
    for w in item.warnings:
        if w.sub_criterion_id:
            existing_by_sub.setdefault(w.sub_criterion_id, []).append(w)

    subs_by_id = {sc.id: sc for sc in item.sub_criteria}
    for sub_id, text in flagged.items():
        sub = subs_by_id.get(sub_id)
        if sub is None or sub.status not in (Status.NOT_APPLICABLE, Status.CANNOT_DETERMINE):
            continue
        existing = existing_by_sub.get(sub_id)
        if existing:
            for w in existing:
                if w.source == "llm" and w.severity == "low":
                    w.severity = "medium"
            continue
        item.warnings.append(ReviewWarning(
            sub_criterion_id=sub_id,
            severity="medium",
            message=f"Not confirmed on the call: {text}. Reviewer should check.",
            evidence=list(sub.evidence),
            source="rule",
        ))
