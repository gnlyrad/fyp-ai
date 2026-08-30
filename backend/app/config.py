"""Runtime configuration for local development and the Azure production path."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Read provider settings at application start without embedding secrets in code."""

    llm_provider: str
    retrieval_provider: str
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: int
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_chat_deployment: str
    azure_openai_embedding_deployment: str
    azure_openai_api_version: str
    azure_search_endpoint: str
    azure_search_api_key: str
    azure_search_index: str
    azure_search_api_version: str
    azure_search_semantic_configuration: str
    database_path: str
    regulatory_context_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_provider=os.getenv("CLEARCALL_LLM_PROVIDER", "ollama").lower(),
            retrieval_provider=os.getenv("CLEARCALL_RETRIEVAL_PROVIDER", "local").lower(),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            ollama_timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120")),
            azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            azure_openai_api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            azure_openai_chat_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", ""),
            azure_openai_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", ""),
            azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT", ""),
            azure_search_api_key=os.getenv("AZURE_SEARCH_API_KEY", ""),
            azure_search_index=os.getenv("AZURE_SEARCH_INDEX", ""),
            azure_search_api_version=os.getenv("AZURE_SEARCH_API_VERSION", "2025-09-01"),
            azure_search_semantic_configuration=os.getenv("AZURE_SEARCH_SEMANTIC_CONFIGURATION", ""),
            database_path=os.getenv("CLEARCALL_JOB_DB_PATH", "artifacts/clearcall_jobs.sqlite3"),
            regulatory_context_path=os.getenv("REGULATORY_CONTEXT_PATH", ""),
        )

    @property
    def azure_openai_configured(self) -> bool:
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key and self.azure_openai_chat_deployment)

    @property
    def azure_search_configured(self) -> bool:
        return bool(self.azure_search_endpoint and self.azure_search_api_key and self.azure_search_index)
