"""The regulatory knowledge base, exposed as a tool.

Backed by the configured FAISS or S3 Vectors index in `langchain-backend`.
This module is the only
place the two backends touch, so swapping FAISS for OpenSearch Serverless or a
Bedrock Knowledge Base later means editing `RegulationSearch`, not this graph.

Written as a `@tool` rather than a plain function so it shows up as its own run
in LangSmith and can be exercised on its own. It is still *called by the graph*,
never chosen by the model — the guidance every checklist item receives has to be
the same on every run for a finding to be reproducible.
"""
from __future__ import annotations

import os
from functools import lru_cache

from langchain_core.tools import tool

from .state import RegulationPassage

GUIDANCE_TOP_K = int(os.getenv("CLEARCALL_GUIDANCE_TOP_K", "3"))
# The cache is a real saving, but a cache hit skips the retriever entirely, so
# the `regulation_search` step is missing from that run's LangSmith trace. Set
# CLEARCALL_GUIDANCE_CACHE=0 when you want every run traced end to end.
CACHE_ENABLED = os.getenv("CLEARCALL_GUIDANCE_CACHE", "1") != "0"

_search = None


def _regulation_search():
    """The shared `RegulationSearch`, built on first use.

    Imported lazily so that a checkout without `langchain-backend` installed can
    still import the graph; the missing dependency then degrades to "no
    guidance" at call time rather than breaking import of the whole module.
    """
    global _search
    if _search is None:
        from langchain_backend.retrieval import RegulationSearch

        _search = RegulationSearch()
    return _search


@lru_cache(maxsize=256)
def _cached_search(query: str, k: int) -> tuple[RegulationPassage, ...]:
    """Cached because the checklist is static: `suitability` asks the same thing
    every run, so the corpus answer is the same too. Without this, every run
    spends one Bedrock embedding call per guided checklist item.

    Returns a tuple because `lru_cache` hands the same object to every caller,
    so it must not be something a caller can mutate.
    """
    documents = _regulation_search().search(query, k=k)
    passages = []
    for document in documents:
        metadata = document.metadata
        page = metadata.get("page")
        passages.append({
            "text": document.page_content,
            "source": metadata.get("source", metadata.get("path", "unknown")),
            # PyPDFLoader counts from zero; analysts read the printed page.
            "page": page + 1 if isinstance(page, int) else None,
        })
    return tuple(passages)


@tool
def search_regulations(query: str, k: int = GUIDANCE_TOP_K) -> list[RegulationPassage]:
    """Find the MAS notice and guideline passages that define the regulatory
    standard for a compliance question.

    Returns extracts with their source document and page. These describe what
    the regulator requires; they are not evidence about any particular call.
    """
    try:
        if not CACHE_ENABLED:
            return list(_cached_search.__wrapped__(query, k))
        return list(_cached_search(query, k))
    except Exception:
        # An explicitly selected cloud backend must not silently produce a
        # report without guidance when credentials/configuration are broken.
        if os.getenv("REGULATION_SEARCH_BACKEND", "faiss").strip().lower() == "s3vectors":
            raise
        # The corpus is an enhancement, not a dependency. A missing index,
        # absent AWS credentials or a throttled embedding call must leave the
        # pipeline judging exactly as it did before guidance existed.
        return []
