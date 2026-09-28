"""`role_inference`: map speaker labels to Advisor / Client.

Three sources, cheapest first (CLAUDE.md "role_inference" section):

1. Caller-supplied `speaker_roles` override on state -- validated against
   Advisor/Client (RM -> Advisor, Customer -> Client, case-insensitive).
   Unknown speaker keys are ignored and recorded in `errors`.
2. Dual-channel recordings: `ch_0`/`ch_1` mapped in code via the
   `COMPLIANCE_CH0_ROLE` env var (default "Advisor"); the other channel
   gets the remaining role. No LLM.
3. Otherwise, one LLM call over the first `MAX_SEGMENTS` segments for any
   speaker still unresolved after (1) and (2).

Roles are exactly "Advisor" / "Client"; `None` means unresolved. Never
guess a role from speaker order or fill in the "other" speaker of a pair
just because one was resolved -- see `prompts/role_inference.py`.
"""
from __future__ import annotations

import os

from compliance_pipeline.llm import get_llm_response
from compliance_pipeline.nodes._common import format_transcript
from compliance_pipeline.prompts.role_inference import SYSTEM, build_user
from compliance_pipeline.schemas import RoleInferenceOutput
from compliance_pipeline.state import ComplianceState

MAX_SEGMENTS = 12
MAX_CHARS_PER_SEGMENT = 600
MIN_ROLE_CONFIDENCE = 0.6

ROLES = {"Advisor", "Client"}
_ROLE_ALIASES = {
    "advisor": "Advisor",
    "rm": "Advisor",
    "client": "Client",
    "customer": "Client",
}


def _normalise_role(value) -> str | None:
    """Map a caller-supplied role string (incl. RM/Customer aliases) to
    Advisor/Client, case-insensitively. Returns None if not recognised."""
    if not isinstance(value, str):
        return None
    return _ROLE_ALIASES.get(value.strip().lower())


def _distinct_speakers(segments: list[dict]) -> list[str]:
    seen: list[str] = []
    for seg in segments:
        speaker = seg.get("speaker")
        if speaker and speaker not in seen:
            seen.append(speaker)
    return seen


def role_inference_node(state: ComplianceState) -> dict:
    segments = state.get("segments") or []
    if not segments:
        return {}

    speakers = _distinct_speakers(segments)
    resolved: dict[str, str] = {}
    source: dict[str, str] = {}
    errors: list[str] = []

    # 1. Caller override.
    caller_roles = state.get("speaker_roles") or {}
    for speaker, raw_role in caller_roles.items():
        if speaker not in speakers:
            errors.append(f"role_inference: unknown speaker in speaker_roles: {speaker!r}")
            continue
        role = _normalise_role(raw_role)
        if role is None:
            errors.append(f"role_inference: invalid role for speaker {speaker!r}: {raw_role!r}")
            continue
        resolved[speaker] = role
        source[speaker] = "caller"

    # 2. Dual channel: map in code, no LLM.
    if state.get("dual_channel"):
        ch0_role = os.environ.get("COMPLIANCE_CH0_ROLE", "Advisor")
        if ch0_role not in ROLES:
            ch0_role = "Advisor"
        ch1_role = "Client" if ch0_role == "Advisor" else "Advisor"
        channel_roles = {"ch_0": ch0_role, "ch_1": ch1_role}
        for speaker in speakers:
            if speaker in resolved or speaker not in channel_roles:
                continue
            resolved[speaker] = channel_roles[speaker]
            source[speaker] = "channel"

    # 3. LLM for whatever's left, unless there's nothing to disambiguate.
    unresolved = [s for s in speakers if s not in resolved]
    if unresolved and len(speakers) > 1:
        prompt_segments = []
        for seg in segments[:MAX_SEGMENTS]:
            seg = dict(seg)
            seg["text"] = (seg.get("text") or "")[:MAX_CHARS_PER_SEGMENT]
            prompt_segments.append(seg)

        result = get_llm_response(
            "role_inference",
            SYSTEM,
            build_user(prompt_segments, unresolved),
            RoleInferenceOutput,
        )
        if not result.ok:
            errors.append(f"role_inference: {result.error}")
        else:
            accepted: dict[str, tuple[str, float]] = {}
            for guess in result.value.speakers:
                if guess.speaker not in unresolved:
                    continue
                if guess.role == "uncertain":
                    continue
                if guess.confidence < MIN_ROLE_CONFIDENCE:
                    continue
                if not guess.reason.strip():
                    continue
                accepted[guess.speaker] = (guess.role, guess.confidence)

            # Consistency on a 2-speaker call: the two speakers cannot share a
            # role. A caller- or channel-assigned role always wins over an LLM
            # guess; between two LLM guesses the more confident one wins. The
            # loser is left unresolved, never flipped to the other role.
            if len(speakers) == 2:
                for speaker, (role, _conf) in list(accepted.items()):
                    holder = next((s for s, r in resolved.items() if r == role), None)
                    if holder is not None:
                        del accepted[speaker]
                        errors.append(
                            f"role_inference: dropped LLM guess {speaker}={role}; "
                            f"conflicts with {source[holder]}-assigned {holder}={role}"
                        )
                if len(accepted) == 2 and len({role for role, _ in accepted.values()}) == 1:
                    winner, loser = sorted(accepted, key=lambda sp: accepted[sp][1], reverse=True)
                    errors.append(
                        f"role_inference: both speakers guessed as {accepted[winner][0]}; "
                        f"kept {winner}, left {loser} unresolved"
                    )
                    del accepted[loser]

            for speaker, (role, _confidence) in accepted.items():
                resolved[speaker] = role
                source[speaker] = "llm"

    new_segments = []
    for seg in segments:
        seg = dict(seg)
        speaker = seg.get("speaker")
        seg["role"] = resolved.get(speaker)
        seg["role_source"] = source.get(speaker)
        new_segments.append(seg)

    transcript = format_transcript(new_segments, with_timestamps=True)

    out: dict = {
        "segments": new_segments,
        "speaker_roles": dict(resolved),
        "transcript": transcript,
    }
    if errors:
        out["errors"] = errors
    return out
