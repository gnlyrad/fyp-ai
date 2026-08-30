"""Answer-generator adapters for local Ollama and Azure OpenAI."""
from __future__ import annotations

import json
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import Settings


class LLMUnavailableError(RuntimeError):
    """Raised when the configured local LLM cannot be reached or decoded."""


class OllamaLLM:
    """Minimal Ollama chat client with structured JSON output."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()

    def analyse(self, question: str, context: str) -> dict[str, str]:
        """Return a schema-constrained assessment based solely on ``context``."""
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            # ``json`` works with a wider range of local Ollama/model versions
            # than full JSON-schema mode. The RAG layer validates every field.
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a financial-call compliance evidence analyser. "
                        "Use only the supplied transcript context. Do not infer facts. "
                        "Pass means the question is clearly satisfied; Fail means the "
                        "context clearly shows it was not satisfied; Unclear means the "
                        "context is insufficient. evidence_quote must be an exact, short "
                        "quote copied from the context. Return exactly one JSON object "
                        "with these string keys and no others: verdict, evidence_quote, "
                        "reason. verdict must be exactly Pass, Fail, or Unclear."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Transcript context:\n{context}\n\nQuestion: {question}",
                },
            ],
        }
        request = Request(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.settings.ollama_timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["message"]["content"]
            answer = json.loads(content)
        except (URLError, TimeoutError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LLMUnavailableError(
                f"Could not get a valid response from Ollama model '{self.settings.ollama_model}'. "
                "Confirm Ollama is running and the model is installed."
            ) from exc

        if not isinstance(answer, dict):
            raise LLMUnavailableError("Ollama returned an invalid structured response.")
        return {key: str(answer.get(key, "")).strip() for key in ("verdict", "evidence_quote", "reason")}


class AzureOpenAILLM:
    """Azure OpenAI chat-completions adapter for the production answer generator."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        if not self.settings.azure_openai_configured:
            raise LLMUnavailableError("Azure OpenAI is not configured. Set endpoint, API key and chat deployment.")

    def analyse(self, question: str, context: str) -> dict[str, str]:
        payload = {
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a financial-call compliance evidence analyser. Use only the supplied "
                        "context. Return JSON with verdict, evidence_quote and reason. verdict must "
                        "be Pass, Fail, Unclear, or N/A. evidence_quote must be copied exactly."
                    ),
                },
                {"role": "user", "content": f"Transcript and regulatory context:\n{context}\n\nQuestion: {question}"},
            ],
        }
        url = (
            f"{self.settings.azure_openai_endpoint.rstrip('/')}/openai/deployments/"
            f"{self.settings.azure_openai_chat_deployment}/chat/completions?api-version="
            f"{self.settings.azure_openai_api_version}"
        )
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "api-key": self.settings.azure_openai_api_key},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            answer = json.loads(body["choices"][0]["message"]["content"])
        except (URLError, TimeoutError, OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise LLMUnavailableError("Azure OpenAI did not return a valid structured answer.") from exc
        return {key: str(answer.get(key, "")).strip() for key in ("verdict", "evidence_quote", "reason")}


def build_answer_generator(settings: Settings | None = None) -> OllamaLLM | AzureOpenAILLM:
    """Select the configured answer-generator provider."""
    active_settings = settings or Settings.from_env()
    if active_settings.llm_provider == "azure_openai":
        return AzureOpenAILLM(active_settings)
    if active_settings.llm_provider == "ollama":
        return OllamaLLM(active_settings)
    raise LLMUnavailableError(f"Unsupported CLEARCALL_LLM_PROVIDER: {active_settings.llm_provider}")
