"""Gemini Developer API clients shared by chat and both retrieval paths."""
from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from .settings import settings


def _api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key or not key.strip():
        raise ValueError("Set GOOGLE_API_KEY in fyp-ai/.env to use Gemini.")
    return key.strip()


def create_chat_model(model_id=None, **options):
    configuration = dict(temperature=settings.temperature, timeout=60, max_retries=0)
    configuration.update(options)
    return ChatGoogleGenerativeAI(
        model=model_id or settings.chat_model, api_key=_api_key(), vertexai=False,
        **configuration,
    )


def create_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=_api_key(), vertexai=False,
    )
