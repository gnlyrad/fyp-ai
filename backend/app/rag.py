"""Retrieval, augmentation and evidence-grounded local LLM orchestration.

This module is the local implementation of the RAG layer.  Lexical retrieval is
an intentionally lightweight stand-in for an embedding model plus vector store;
the LLM boundary is isolated in :mod:`app.llm` for a later SageMaker endpoint.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import Settings
from .llm import LLMUnavailableError, build_answer_generator
from .search import AzureAISearchRetriever, RetrievalUnavailableError

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z'-]+")
STOP_WORDS = {"the", "a", "an", "and", "or", "to", "of", "for", "in", "on", "is", "was", "are", "i", "you", "we", "it", "that", "this", "with", "my", "your"}


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    start_seconds: int
    end_seconds: int
    score: float


def tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text) if token.lower() not in STOP_WORDS]


def format_timestamp(seconds: int) -> str:
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def normalise_verdict(value: str) -> str | None:
    """Accept only the three verdicts exposed by the API."""
    choices = {"pass": "Pass", "fail": "Fail", "unclear": "Unclear", "n/a": "N/A", "na": "N/A", "not applicable": "N/A"}
    return choices.get(value.strip().lower())


def build_chunks(utterances: list[dict], window_size: int = 4) -> list[dict]:
    chunks: list[dict] = []
    for index in range(0, len(utterances), window_size):
        window = utterances[index:index + window_size]
        chunks.append({
            "chunk_id": f"chunk-{index // window_size + 1:02d}",
            "start_seconds": window[0]["start_seconds"],
            "end_seconds": window[-1]["end_seconds"],
            "text": " ".join(f'{item["speaker"]}: {item["text"]}' for item in window),
        })
    return chunks


def retrieve(question: str, chunks: Iterable[dict], top_k: int = 3) -> list[RetrievedChunk]:
    query = Counter(tokens(question))
    results: list[RetrievedChunk] = []
    for chunk in chunks:
        document = Counter(tokens(chunk["text"]))
        overlap = sum(min(query[token], document[token]) for token in query)
        # A lightweight lexical score with a small preference for concise chunks.
        score = overlap / max(1, len(query)) + (0.05 / max(1, len(document)))
        results.append(RetrievedChunk(score=round(score, 4), **chunk))
    return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]


class ComplianceRAGWorkflow:
    """Explicit retrieve -> relevance check -> augment -> generate workflow.

    Locally this uses lexical retrieval and Ollama. In the deployed architecture,
    set the provider variables to use Azure AI Search hybrid retrieval and Azure
    OpenAI answer generation without changing API callers.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.answer_generator = build_answer_generator(self.settings)
        self.azure_retriever = AzureAISearchRetriever(self.settings) if self.settings.retrieval_provider == "azure_ai_search" else None

    def retrieve_evidence(self, question: str, chunks: list[dict]) -> list[RetrievedChunk]:
        if self.azure_retriever:
            return [RetrievedChunk(**item) for item in self.azure_retriever.search(question)]
        return retrieve(question, chunks)

    def retrieve_regulatory_context(self, question: str) -> str:
        if self.azure_retriever:
            results = self.azure_retriever.search(question, top_k=2, filter_expression="source_type eq 'regulatory'")
            return "\n\n".join(item["text"] for item in results)
        if not self.settings.regulatory_context_path:
            return ""
        path = Path(self.settings.regulatory_context_path)
        if not path.exists():
            return ""
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(content, list):
                return "\n\n".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content)
        except json.JSONDecodeError:
            pass
        return path.read_text(encoding="utf-8")

    def analyse(self, item: dict, chunks: list[dict]) -> dict:
        """Run the RAG state sequence and return a validated API result."""
        retrieved = self.retrieve_evidence(item["question"], chunks)
        transcript_context = "\n\n".join(
            f"[{format_timestamp(result.start_seconds)}-{format_timestamp(result.end_seconds)}] {result.text}"
            for result in retrieved
        )
        regulatory_context = self.retrieve_regulatory_context(item["question"])
        augmented_context = transcript_context
        if regulatory_context:
            augmented_context += f"\n\nRegulatory context (guidance only):\n{regulatory_context}"
        try:
            answer = self.answer_generator.analyse(item["question"], augmented_context)
            verdict = normalise_verdict(answer["verdict"])
            evidence_quote = answer["evidence_quote"]
            # Relevance gate: cited evidence must occur in retrieved transcript
            # evidence, never merely in regulatory guidance or model prose.
            matched = next(
                (result for result in retrieved if evidence_quote and evidence_quote.lower() in result.text.lower()),
                None,
            )
            supported = matched is not None
            if not verdict or not supported:
                verdict = "Unclear"
                reason = "Insufficient retrieved evidence to support a definitive outcome."
            else:
                reason = answer["reason"] or "Assessment is supported by the quoted transcript evidence."
        except (LLMUnavailableError, RetrievalUnavailableError) as exc:
            verdict = "Unclear"
            evidence_quote = ""
            matched = None
            supported = False
            reason = str(exc)

        return {
            "checklist_id": item["id"],
            "question": item["question"],
            "verdict": verdict,
            "reason": reason,
            "confidence": "high" if supported and verdict not in {"Unclear", "N/A"} else "low",
            "evidence_quote": evidence_quote if supported else None,
            "timestamp": format_timestamp(matched.start_seconds) if matched and supported else None,
            "chunk_id": matched.chunk_id if matched and supported else None,
            "retrieved_evidence": [result.__dict__ for result in retrieved],
            "regulatory_context_used": bool(regulatory_context),
            "retrieval_provider": self.settings.retrieval_provider,
            "generation_provider": self.settings.llm_provider,
        }


def analyse_checklist_item(item: dict, chunks: list[dict], workflow: ComplianceRAGWorkflow | None = None) -> dict:
    """Compatibility wrapper for the FastAPI API and offline evaluation.

    ``expected_verdict``, ``reason`` and dataset evidence labels are deliberately
    never sent to the LLM. They remain ground truth for offline evaluation only.
    """
    try:
        return (workflow or ComplianceRAGWorkflow()).analyse(item, chunks)
    except (LLMUnavailableError, RetrievalUnavailableError) as exc:
        return {
            "checklist_id": item["id"], "question": item["question"], "verdict": "Unclear",
            "reason": str(exc), "confidence": "low", "evidence_quote": None,
            "timestamp": None, "chunk_id": None, "retrieved_evidence": [],
            "regulatory_context_used": False, "retrieval_provider": "unavailable", "generation_provider": "unavailable",
        }
