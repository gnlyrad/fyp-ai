"""Gemini provider contracts; no API calls or real credentials."""
import os
from unittest.mock import patch

import pytest
from google.genai.errors import ClientError, ServerError

from langchain_backend import models
from langchain_backend.settings import Settings
from compliance_pipeline.reliability import error_details


def test_clients_use_api_key_and_never_vertex_or_aws_credentials():
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key", "GOOGLE_GENAI_USE_VERTEXAI": "true",
                                 "AWS_ACCESS_KEY_ID": "unused"}, clear=True), \
         patch.object(models, "ChatGoogleGenerativeAI") as chat, \
         patch.object(models, "GoogleGenerativeAIEmbeddings") as embeddings:
        models.create_chat_model()
        models.create_embeddings()
    assert chat.call_args.kwargs["api_key"] == "test-key"
    assert embeddings.call_args.kwargs["google_api_key"] == "test-key"
    for factory in (chat, embeddings):
        assert factory.call_args.kwargs["vertexai"] is False
        assert "region_name" not in factory.call_args.kwargs


def test_missing_google_key_does_not_fall_back_to_aws():
    with patch.dict(os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "unused"}, clear=True):
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            models.create_chat_model()
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            models.create_embeddings()


def test_embedding_model_changes_index_namespace(tmp_path):
    with patch.dict(os.environ, {"LANGCHAIN_INDEX_DIR": str(tmp_path),
                                 "GOOGLE_EMBEDDING_MODEL": "model-a"}):
        first = Settings().index_dir
        with patch.dict(os.environ, {"GOOGLE_EMBEDDING_MODEL": "model-b"}):
            second = Settings().index_dir
    assert first != second
    assert first.parent == second.parent == tmp_path
    assert first != tmp_path


@pytest.mark.parametrize("code,retryable", [(400, False), (401, False), (403, False),
                                         (404, False), (429, True), (500, True), (503, True)])
def test_google_errors_keep_status_through_wrappers(code, retryable):
    original = (ClientError if code < 500 else ServerError)(
        code, {"error": {"code": code, "message": "sensitive request"}})
    wrapper = RuntimeError("provider wrapper")
    wrapper.__cause__ = original
    detail, malformed = error_details(wrapper)
    assert detail["error_code"] == str(code)
    assert detail["retryable"] is retryable
    assert not malformed
    assert "sensitive request" not in str(detail)
