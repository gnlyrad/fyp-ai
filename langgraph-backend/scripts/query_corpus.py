"""Query the MAS corpus directly — retrieval only, no LLM, no graph.

The fastest way to judge whether a bad verdict is a retrieval problem or a
judgement problem: run the checklist item's query here and read what comes back.
Scores are cosine distance (LOWER is closer), so a first result above ~0.9 means
nothing in the corpus really matches.

    python scripts/query_corpus.py "cooling off cancellation period"
    python scripts/query_corpus.py --item cooling_off
    python scripts/query_corpus.py --all          # every checklist item, one line each
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
os.environ.update({k: v for k, v in dotenv_values(ROOT / ".env").items() if v})
sys.path.insert(0, str(ROOT / "langgraph-backend"))

from compliance_pipeline.nodes.guidance import guidance_query
from compliance_pipeline.prompts.checklist import CHECKLIST, CHECKLIST_BY_ID


def page_of(meta) -> str:
    page = meta.get("page")
    return str(page + 1) if isinstance(page, int) else "?"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="free-text query")
    parser.add_argument("--item", help="use a checklist item's query instead")
    parser.add_argument("--all", action="store_true", help="one line per checklist item")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    from langchain_backend.retrieval import RegulationSearch

    search = RegulationSearch()
    store = search.vector_store()

    if args.all:
        print(f"top-1 match per checklist item (k=1, lower score = closer)\n")
        for item in CHECKLIST:
            hit = store.similarity_search_with_score(guidance_query(item), k=1)
            if not hit:
                print(f"  {item['id']:17} (nothing)")
                continue
            doc, score = hit[0]
            src = doc.metadata.get("source", "?")
            print(f"  {item['id']:17} {score:.3f}  {src[:52]:52} p.{page_of(doc.metadata)}")
        return

    if args.item:
        item = CHECKLIST_BY_ID.get(args.item)
        if item is None:
            sys.exit(f"no checklist item '{args.item}'. ids: "
                     + ", ".join(sorted(CHECKLIST_BY_ID)))
        query = guidance_query(item)
        print(f"item  : {args.item} ({item.get('question_type')})")
    else:
        query = " ".join(args.query)
        if not query:
            sys.exit("give a query, or --item <id>, or --all")

    print(f"query : {query}\n")
    for i, (doc, score) in enumerate(store.similarity_search_with_score(query, k=args.k), 1):
        meta = doc.metadata
        print(f"[{i}] score {score:.3f}  {meta.get('source')}  p.{page_of(meta)}")
        print(f"    {' '.join(doc.page_content.split())[:320]}\n")


if __name__ == "__main__":
    main()
