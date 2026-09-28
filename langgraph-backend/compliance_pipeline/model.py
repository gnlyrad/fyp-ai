"""The single Gemini chat model shared by every node that calls an LLM.

Both classification and analysis talk to the same model. Constructing it once
here â€” rather than at import time in each node â€” keeps one client, one place to
change the model id, and one place where `.env` is guaranteed to have been
loaded before the Gemini client reads the credentials.
"""
from __future__ import annotations

from langchain_backend.models import create_chat_model
from langchain_backend.settings import settings

MODEL_ID = settings.chat_model

_model = None


def get_model():
    """The shared chat model, built on first use."""
    global _model
    if _model is None:
        _model = create_chat_model()
    return _model
