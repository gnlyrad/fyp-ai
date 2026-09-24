"""Cached rule lookup per checklist item.

Rules per item are fixed, so the lookup is built once (`build_rule_cache`,
run offline via `scripts/build_rule_cache.py`) and read at runtime
(`get_rules`, cheap and lru_cached). `check_item` never calls the vector
store itself.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_PATH = DATA_DIR / "rule_cache.json"

# One query per red-flag category, in addition to whatever rule_queries C06
# carries in checklist.yaml (kept in sync there; see build_rule_cache).
RULES_PER_QUERY = 3


def _rule_id(source: str, page, text: str) -> str:
    digest = hashlib.sha1(f"{source}|{page}|{text}".encode("utf-8")).hexdigest()
    return digest[:12]


@lru_cache(maxsize=1)
def _load_cache() -> dict[str, list[dict]]:
    if not CACHE_PATH.exists():
        return {}
    return json.loads(CACHE_PATH.read_text(encoding="utf-8"))


def get_rules(item_id: str) -> list[dict]:
    """Cached rule chunks for `item_id`: `[{rule_id, text, source, page}]`.

    Missing cache file (not yet built) -> `[]`; the check still runs, just
    without cited rule text.
    """
    return _load_cache().get(item_id, [])


def build_rule_cache() -> dict[str, list[dict]]:
    """Run every checklist item's `rule_queries` against the regulation search
    and write the result to `CACHE_PATH`. C06 runs one query per red-flag
    category (its `rule_queries` list in checklist.yaml already has six).

    Imports `langchain_backend` lazily so importing `compliance.retrieval`
    never requires AWS credentials or the S3 Vectors index.
    """
    from langchain_backend.retrieval import RegulationSearch  # lazy

    from compliance.config import load_checklist

    search = RegulationSearch()
    cache: dict[str, list[dict]] = {}
    for item in load_checklist():
        seen: set[str] = set()
        chunks: list[dict] = []
        for query in item.rule_queries:
            for doc in search.search(query, k=RULES_PER_QUERY):
                source = doc.metadata.get("source", "")
                page = doc.metadata.get("page")
                rule_id = _rule_id(source, page, doc.page_content)
                if rule_id in seen:
                    continue
                seen.add(rule_id)
                chunks.append({
                    "rule_id": rule_id,
                    "text": doc.page_content,
                    "source": source,
                    "page": page,
                })
        cache[item.id] = chunks

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    _load_cache.cache_clear()
    return cache
