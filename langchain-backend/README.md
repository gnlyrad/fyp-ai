# LangChain Backend

This folder is a starter RAG backend for answering questions over a PDF corpus using Gemini.

## What it does

- Loads PDFs from a dedicated corpus folder.
- Splits them into retrieval-sized chunks.
- Embeds the chunks with Gemini embeddings.
- Stores the chunks in a local FAISS index by default.
- Can explicitly ingest the PDF corpus into S3 Vectors; retrieval still uses FAISS.
- Uses an Gemini chat model to answer questions from retrieved context.

## Retrieval flow

```mermaid
flowchart LR
  PDFs[PDF corpus folder] --> Load[PDF loading]
  Load --> Chunk[Chunking]
  Chunk --> Embed[Gemini embeddings]
  Embed --> Store[(FAISS now / OpenSearch Serverless later)]
  Ask[User question] --> QEmbed[Question embedding]
  QEmbed --> Search[Semantic search]
  Search --> Store
  Store --> Context[Retrieved code context]
  Context --> Gemini[Gemini LLM]
  Gemini --> Answer[Grounded answer + sources]
```

## Gemini configuration

Set `GOOGLE_API_KEY` in the repository-root `fyp-ai/.env`.

- Chat: `gemini-3.5-flash` (`CLEARCALL_MODEL_ID`).
- Embeddings: `gemini-embedding-001` (`GOOGLE_EMBEDDING_MODEL`), 3072 dimensions.
- Temperature: 1 (`CLEARCALL_TEMPERATURE`).
- Authentication: Gemini Developer API key, with Vertex AI disabled.

Local FAISS indexes live under `.index/google-<model hash>/faiss`.
The first retrieval builds the selected model's index from the PDF corpus.
Changing embedding models creates a separate index; existing Bedrock vectors
are not reused or overwritten. Normal chat and retrieval require no AWS credentials.

## Endpoints

- `POST /ingest` builds or refreshes the local FAISS index.
- `POST /ingest?target=s3vectors` uploads corpus PDF chunks to S3 Vectors.
- Add `&dry_run=true` to preview S3 ingestion without embedding or uploading.
- `POST /ask` answers a question using retrieved code context.
- `GET /health` returns a simple status payload.

## Corpus folder

Put your PDFs in [corpus/](corpus/) under this folder. The ingestion step reads every `*.pdf` file recursively from that directory.

## Local setup

1. Create a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Set `GOOGLE_API_KEY` in `fyp-ai/.env`.
4. Run the app with `python -m langchain_backend.main`.

## S3 Vectors ingestion only

Only PDFs inside the configured corpus folder (including nested folders) are
loaded. Checklist Python/JSON files, transcripts, and files outside the corpus
are not ingested. Linked files resolving outside the corpus are skipped.
The PDF text is split into 1,200-character chunks with 200-character overlap,
embedded with Gemini, and uploaded with text, relative source filename,
zero-based page, embedding model, and corpus version metadata.

This is an explicit upload path. `/ask` and LangGraph guidance still query local
FAISS. Their automatic local index creation does not trigger an S3 upload.

### Preview without an AWS bucket

From `langchain-backend`, using the environment where dependencies are installed:

```powershell
python -m langchain_backend.s3_ingestion --dry-run
```

This does not construct AWS clients or invoke Gemini. The response reports
`pdf_files`, `documents` (PDF pages, preserving the existing API meaning),
`chunks`, `corpus_version`, and `vectors_uploaded: 0`. PDFs without extractable
text require OCR before they can contribute chunks.

### Create the AWS destination once

This optional storage-only path still needs an AWS CLI/profile. Your Google
API key authenticates embeddings; it does not authenticate S3 Vectors requests.
The following example creates a dedicated vector bucket and an index compatible
with the default `gemini-embedding-001` model. Substitute your own names:

```powershell
aws s3vectors create-vector-bucket --vector-bucket-name clearcall-mas-corpus --region ap-southeast-1
aws s3vectors create-index --vector-bucket-name clearcall-mas-corpus --index-name mas-gemini-001 --data-type float32 --dimension 3072 --distance-metric cosine --metadata-configuration nonFilterableMetadataKeys=text --region ap-southeast-1
```

Add the destination to the repository-root `fyp-ai/.env`:

```dotenv
AWS_REGION=ap-southeast-1
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
S3_VECTORS_BUCKET_NAME=clearcall-mas-corpus
S3_VECTORS_INDEX_NAME=mas-gemini-001
```

Alternatively set `S3_VECTORS_INDEX_ARN` instead of the bucket/index pair; the ARN
takes precedence. Use an index in `AWS_REGION`. Provide standard AWS credentials
through an AWS profile, environment variables, or the application's IAM role.
The ingestion identity needs `s3vectors:GetIndex` and `s3vectors:PutVectors` on
the destination index, plus a Google API key for embeddings. Provisioning
also needs `s3vectors:CreateVectorBucket` and `s3vectors:CreateIndex`. If you use a
customer-managed KMS key, its policy and permissions must allow the operation.

### Upload

```powershell
python -m langchain_backend.s3_ingestion
```

Or, with the existing API running:

```powershell
Invoke-RestMethod -Method Post -Uri 'http://localhost:8001/ingest?target=s3vectors'
```

From Python, the existing service also supports
`RegulationSearch().ingest(target="s3vectors")`. The standalone command avoids
initializing the chat model and is recommended for ingestion jobs.

The writer verifies the index configuration before embedding and checks vector
dimensions before uploading each batch. It sends at most 64 vectors per request.
AWS errors propagate rather than reporting success. A failure can leave earlier
batches uploaded; rerunning an unchanged corpus reuses the same keys. It still
regenerates embeddings, so retrying incurs embedding charges.

Ingestion is versioned and does not delete old vectors. Revised PDFs produce a
new `corpus_version` and new keys. For a replacement corpus, use a fresh index,
or have future S3 retrieval filter by the version returned by a **successful**
ingestion. Do not query an incomplete version after a failed run. This change
does not provision resources, upload the original PDF binaries, or implement S3
query retrieval.

### Offline verification

```powershell
python -m unittest discover -s tests -v
```

Tests load real PDF fixtures and validate S3 request shapes using Botocore's
Stubber, with mocked embeddings. No AWS access or charges are required.
