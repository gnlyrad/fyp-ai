"""`role_inference`: map speaker labels to Advisor / Client.

Final intended behaviour (CLAUDE.md): one cheap LLM call on the first ~12
segments, mapping each speaker label to Advisor / Client / uncertain and
writing `role` onto segments. Skipped when the recording is dual-channel
(channel map done in code) or when the caller already passed
`speaker_roles`.

Step-0 stub: no-op, segments keep `role: None`.
"""
from __future__ import annotations

from compliance.state import ComplianceState


def role_inference_node(state: ComplianceState) -> dict:
    return {}
