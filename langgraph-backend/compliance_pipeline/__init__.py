"""CLAUDE.md-spec compliance call review pipeline (see CLAUDE.md).

Implements the staged pipeline described in `langgraph-backend/CLAUDE.md`:
ingest -> quality_check -> role_inference -> call_intent -> gate ->
Stage 2 checks (Send fan-out) -> verify -> decide -> aggregate -> output.

`ingest` can fetch the AWS Transcribe output itself from an S3 URL
(`nodes/s3_fetch.py`).
"""
from __future__ import annotations
