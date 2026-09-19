# ClearCall compliance pipeline (LangGraph)

Judges a call transcript against a compliance checklist, grounded in the MAS
regulatory corpus held by `../langchain-backend`.

```
ingestion -> classification -> guidance -> retrieval -> (fan-out) analysis -> output
```

`guidance` looks up the regulatory standard for each `targeted` and `exhaustive`
checklist item; `deterministic` presence checks are deliberately excluded (see
`nodes/guidance.py` for why). Every finding carries a `regulatory_basis` naming
the notice and page it was judged under.

## Setup (once)

From the repo root:

```powershell
python -m pip install -r langgraph-backend/requirements.txt
```

That pulls in `langchain-backend` as an editable install, which is what supplies
the corpus and its FAISS index.

`.env` at the repo root needs:

```
AWS_BEARER_TOKEN_BEDROCK=...
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=clearcall-compliance
```

Build the vector index once so the first run is not slowed by it:

```powershell
python -c "from langchain_backend.retrieval import RegulationSearch; print(RegulationSearch().ingest())"
```

## Run

```powershell
cd langgraph-backend
langgraph dev
```

Studio opens on `http://127.0.0.1:2024`. To send a request:

```powershell
curl -X POST http://127.0.0.1:2024/runs/wait `
  -H "Content-Type: application/json" `
  -d "@requests/request.json"
```

Omit `checklist` from the input to run the built-in 20-item checklist. Supplying
items as bare `{id, question}` pairs sends them through classification; supplying
`question_type` skips it.

## The regulatory RAG service

`../langchain-backend` also runs standalone, for asking the corpus questions
directly rather than through the pipeline:

```powershell
python -m langchain_backend.main    # http://localhost:8001
```

`POST /ingest` rebuilds the index, `POST /ask` answers from it.

## Tracing

LangChain and LangGraph trace automatically once `LANGSMITH_TRACING=true` and
`LANGSMITH_API_KEY` are set. A pipeline run appears in the
`clearcall-compliance` project as:

```
LangGraph
  ingestion
  classification -> ChatBedrockConverse
  guidance
    search_regulations [tool]
      regulation_search [retriever]     <- embedding + FAISS lookup, with the documents
  retrieval
  analysis_targeted / _deterministic / _exhaustive -> ChatBedrockConverse
  output
```

and an `/ask` call appears as `regulation_rag -> regulation_search + ChatBedrockConverse`.

Guidance results are cached per process, and a cache hit skips the retriever, so
the `regulation_search` step is absent from repeat runs. Set
`CLEARCALL_GUIDANCE_CACHE=0` to trace every run end to end.

## Environment knobs

| Variable | Default | Effect |
|---|---|---|
| `CLEARCALL_MODEL_ID` | `apac.amazon.nova-lite-v1:0` | Judge and classifier model |
| `AWS_REGION` | `ap-southeast-1` | Bedrock region |
| `CLEARCALL_CHUNK_SENTENCES` | `3` | Transcript chunk size |
| `CLEARCALL_TOP_K` | `3` | Chunks per targeted item |
| `CLEARCALL_DETERMINISTIC_TOP_K` | `5` | Chunks per presence check |
| `CLEARCALL_MIN_RELEVANCE` | `0.01` | Floor below which a chunk is not evidence |
| `CLEARCALL_GUIDANCE_TOP_K` | `3` | Regulatory passages per guided item |
| `CLEARCALL_GUIDANCE_CACHE` | `1` | `0` disables the guidance cache |
| `BEDROCK_EMBEDDING_MODEL` | `cohere.embed-english-v3` | Corpus embeddings |
| `LANGCHAIN_INDEX_DIR` | `langchain-backend/.index` | FAISS index location |

## Note on packaging

`langgraph dev` runs in the active virtualenv, so the editable
`../langchain-backend` install resolves. `langgraph build` (Docker) would not —
the parent directory is outside the build context. Vendoring the corpus into
this folder, or publishing `langchain-backend` to a registry, is the fix when
that day comes.
