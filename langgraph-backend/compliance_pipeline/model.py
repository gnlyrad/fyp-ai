"""The single Bedrock chat model shared by every node that calls an LLM.

Both classification and analysis talk to the same model. Constructing it once
here — rather than at import time in each node — keeps one client, one place to
change the model id, and one place where `.env` is guaranteed to have been
loaded before `init_chat_model` reads the credentials.
"""
from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(find_dotenv())  # must run before init_chat_model()

MODEL_ID = os.getenv("CLEARCALL_MODEL_ID", "apac.amazon.nova-lite-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-1")

_model = None


def get_model():
    """The shared chat model, built on first use."""
    global _model
    if _model is None:
        _model = init_chat_model(
            MODEL_ID,
            model_provider="bedrock_converse",
            region_name=AWS_REGION,
            temperature=0,
        )
    return _model
