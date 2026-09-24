"""CLAUDE.md-spec compliance call review pipeline (new package, see CLAUDE.md).

Kept alongside `compliance_pipeline/` (the older per-question design) for an
A/B comparison. This package implements the staged pipeline described in
`langgraph-backend/CLAUDE.md`: ingest -> quality_check -> role_inference ->
Stage 1 facts -> context_lookup -> gate -> Stage 2 checks (Send fan-out) ->
verify -> decide -> aggregate -> output.
"""
from __future__ import annotations
