"""Prove the RAG path actually runs, with observable evidence.

Checks, in order:
  1. the FAISS index exists, and what is in it
  2. a live query embedding really goes to Bedrock (dimensions + latency)
  3. retrieval returns real corpus passages with source and page
  4. the same call, traced, lands in LangSmith as a `regulation_search` run

Run it any time you want to confirm the pipeline is wired end to end:

    python scripts/verify_rag.py
"""
from __future__ import annotations

import collections
import os
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
os.environ.update({k: v for k, v in dotenv_values(ROOT / ".env").items() if v})
# Cache hits skip the retriever, which would hide the very thing being tested.
os.environ["CLEARCALL_GUIDANCE_CACHE"] = "0"

sys.path.insert(0, str(ROOT / "langgraph-backend"))

QUERY = "Was suitability established before a recommendation? risk preference investment horizon objectives"


def rule(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}", flush=True)


def main() -> None:
    from langchain_backend.retrieval import RegulationSearch
    from langchain_backend.settings import settings

    rule("1. INDEX")
    index_dir = settings.index_dir / "faiss"
    print(f"location : {index_dir}")
    print(f"exists   : {index_dir.exists()}")
    if index_dir.exists():
        size = sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file())
        print(f"size     : {size:,} bytes")

    search = RegulationSearch()
    store = search.vector_store()
    docs = list(store.docstore._dict.values())
    print(f"chunks   : {len(docs)}")
    print(f"vectors  : {store.index.ntotal}  (dim {store.index.d})")
    for src, n in collections.Counter(d.metadata.get("source") for d in docs).most_common():
        print(f"           {n:4}  {src}")

    rule("2. EMBEDDING CALL TO BEDROCK")
    print(f"model    : {settings.bedrock_embedding_model}")
    print(f"region   : {settings.aws_region}")
    started = time.perf_counter()
    vector = search._embeddings.embed_query(QUERY)
    elapsed = time.perf_counter() - started
    print(f"query    : {QUERY[:60]}...")
    print(f"returned : {len(vector)} floats in {elapsed * 1000:.0f} ms")
    print(f"sample   : {[round(x, 4) for x in vector[:5]]}")
    print("           (network latency + a real vector = the call left this machine)")

    rule("3. RETRIEVAL")
    started = time.perf_counter()
    results = search.search(QUERY, k=3)
    elapsed = time.perf_counter() - started
    print(f"{len(results)} passages in {elapsed * 1000:.0f} ms\n")
    for i, doc in enumerate(results, start=1):
        meta = doc.metadata
        page = meta.get("page")
        page = page + 1 if isinstance(page, int) else "?"
        print(f"  [{i}] {meta.get('source')}  p.{page}")
        print(f"      {' '.join(doc.page_content.split())[:200]}...\n")

    rule("4. LANGSMITH TRACE")
    tracing = os.environ.get("LANGSMITH_TRACING")
    project = os.environ.get("LANGSMITH_PROJECT")
    print(f"tracing  : {tracing}   project: {project}")
    if tracing != "true":
        print("LANGSMITH_TRACING is not 'true' — nothing was recorded.")
        return

    from langsmith import Client

    client = Client()
    # The retriever run from step 3 is written asynchronously; give it a moment.
    time.sleep(4)
    runs = [
        r for r in client.list_runs(project_name=project, limit=15)
        if r.name == "regulation_search"
    ]
    if not runs:
        print("No `regulation_search` run found yet — traces can lag a few seconds.")
        return

    run = runs[0]
    latency = (run.end_time - run.start_time).total_seconds() if run.end_time else None
    print(f"run name : {run.name}  [{run.run_type}]")
    print(f"run id   : {run.id}")
    print(f"started  : {run.start_time}")
    print(f"latency  : {latency:.2f}s" if latency else "latency  : (pending)")
    print(f"inputs   : {str(run.inputs)[:120]}")
    docs_out = (run.outputs or {}).get("output") or (run.outputs or {}).get("documents")
    print(f"outputs  : {len(docs_out) if isinstance(docs_out, list) else '?'} documents recorded")
    try:
        print(f"url      : {client.get_run_url(run=run)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
