"""The single chat model shared by every node that calls an LLM.

Both classification and analysis talk to the same model. Constructing it once
here — rather than at import time in each node — keeps one client, one place to
change the model id, and one place where `.env` is guaranteed to have been
loaded before `init_chat_model` reads the credentials.

The provider is inferred from the model id so that switching between Bedrock and
Gemini is a one-value change: a `gemini-*` id needs GOOGLE_API_KEY in `.env` and
takes no region, anything else goes to Bedrock in AWS_REGION. Set
CLEARCALL_MODEL_PROVIDER to override the inference.
"""
from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(find_dotenv())  # must run before init_chat_model()

# Anthropic profiles (e.g. apac.anthropic.claude-3-5-sonnet-20241022-v2:0) are a
# better fit for the strict structured-output schemas in nodes/analysis.py, but
# they are gated until the Anthropic use case details form is submitted for this
# AWS account. Nova Pro is the strongest profile currently invokable in APAC.
MODEL_ID = os.getenv("CLEARCALL_MODEL_ID", "apac.amazon.nova-pro-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

MODEL_PROVIDER = os.getenv(
    "CLEARCALL_MODEL_PROVIDER",
    "google_genai" if MODEL_ID.startswith("gemini") else "bedrock_converse",
)

_model = None


def get_model():
    """The shared chat model, built on first use."""
    global _model
    if _model is None:
        # region_name is a Bedrock argument; Gemini is keyed by GOOGLE_API_KEY.
        extra = {"region_name": AWS_REGION} if MODEL_PROVIDER == "bedrock_converse" else {}
        _model = init_chat_model(
            MODEL_ID,
            model_provider=MODEL_PROVIDER,
            temperature=0,
            **extra,
        )
    return _model
