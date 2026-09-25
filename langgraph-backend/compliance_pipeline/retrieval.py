"""Rule lookup per checklist item.

Rules for the default checklist are fixed, so that lookup is built once
(`build_rule_cache`, run offline via `scripts/build_rule_cache.py`) and read
at runtime. A caller-supplied checklist can add items or change an item's
`rule_queries`; those have no cached rules, so `get_rules` searches the
regulation corpus for them live (once per distinct query set per process)
and registers the chunks so `get_rule` can still resolve their ids.
"""
from __future__ import annotations

import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path

from compliance_pipeline.schemas import ChecklistItem

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


def get_rules(item: ChecklistItem | str) -> list[dict]:
    """Rule chunks for a checklist item: `[{rule_id, text, source, page}]`.

    - An item id (str), or an item whose `rule_queries` match the default
      checklist's for that id -> the prebuilt cache. Missing cache file
      (not yet built) -> `[]`; the check still runs, just without rule text.
    - Otherwise (a caller-added item, or edited `rule_queries`) -> a live
      search of the regulation corpus. May raise if the corpus is
      unreachable; `check_item` catches that and runs without rules.
    """
    if isinstance(item, str):
        return _load_cache().get(item, [])

    from compliance_pipeline.config import get_item  # lazy: config <-> retrieval

    default = get_item(item.id)
    cache = _load_cache()
    if default is not None and default.rule_queries == item.rule_queries and item.id in cache:
        return cache[item.id]
    if not item.rule_queries:
        return []
    return list(_live_rules(tuple(item.rule_queries)))


_search = None
_search_lock = threading.Lock()
# rule_id -> chunk for every live-searched rule, so `get_rule` can resolve them.
_live_index: dict[str, dict] = {}


def _regulation_search():
    global _search
    with _search_lock:
        if _search is None:
            from langchain_backend.retrieval import RegulationSearch  # lazy

            _search = RegulationSearch()
    return _search


def _search_chunks(search, queries) -> list[dict]:
    """Run `queries` against `search`, deduped by rule id, in query order."""
    seen: set[str] = set()
    chunks: list[dict] = []
    for query in queries:
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
    return chunks


@lru_cache(maxsize=256)
def _live_rules(queries: tuple[str, ...]) -> tuple[dict, ...]:
    """Live corpus search for one item's queries. Cached per process (the
    corpus doesn't change mid-run); a failure raises and is not cached."""
    chunks = _search_chunks(_regulation_search(), queries)
    for chunk in chunks:
        _live_index.setdefault(chunk["rule_id"], chunk)
    return tuple(chunks)


@lru_cache(maxsize=1)
def _rule_index() -> dict[str, dict]:
    """`rule_id -> {rule_id, text, source, page}` across every checklist
    item's cached chunks, deduplicated (the same regulatory extract can be
    cited by more than one item). Built once from `_load_cache()`."""
    index: dict[str, dict] = {}
    for chunks in _load_cache().values():
        for chunk in chunks:
            rule_id = chunk.get("rule_id")
            if rule_id and rule_id not in index:
                index[rule_id] = chunk
    return index


def get_rule(rule_id: str) -> dict | None:
    """The cached chunk `{rule_id, text, source, page}` for a single rule id,
    or None if it is not in the cache (`output.py`: store rule chunks once
    and reference them by id -- CLAUDE.md "Retrieval")."""
    return _rule_index().get(rule_id) or _live_index.get(rule_id)


def build_rule_cache() -> dict[str, list[dict]]:
    """Run every checklist item's `rule_queries` against the regulation search
    and write the result to `CACHE_PATH`. C06 runs one query per red-flag
    category (its `rule_queries` list in checklist.yaml already has six).

    Imports `langchain_backend` lazily so importing `compliance_pipeline.retrieval`
    never requires AWS credentials or the S3 Vectors index.
    """
    from langchain_backend.retrieval import RegulationSearch  # lazy

    from compliance_pipeline.config import load_checklist

    search = RegulationSearch()
    cache: dict[str, list[dict]] = {}
    for item in load_checklist():
        cache[item.id] = _search_chunks(search, item.rule_queries)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    _load_cache.cache_clear()
    _rule_index.cache_clear()
    return cache
