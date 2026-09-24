# S3 Vectors handoff

S3 retrieval now serves regulatory guidance and `/ask`. Transcript evidence
retrieval stays in memory. FAISS remains the default. No AWS resources are
created automatically and no credentials belong in Git.

## Configure

For the root Docker setup, edit `langgraph-backend/.env` locally:

```dotenv
AWS_REGION=ap-southeast-1
BEDROCK_EMBEDDING_MODEL=cohere.embed-english-v3
REGULATION_SEARCH_BACKEND=s3vectors
S3_VECTORS_BUCKET_NAME=clearcall-mas-corpus
S3_VECTORS_INDEX_NAME=mas-cohere-v3
# Fill after a successful ingestion:
S3_VECTORS_CORPUS_VERSION=
```

Alternatively use `S3_VECTORS_INDEX_ARN`; it overrides the bucket/index pair.
Use a float32, 1024-dimensional cosine index for this embedding model, with
`text` non-filterable and `corpus_version`/`embedding_model` filterable.
Supply standard AWS credentials to the container through environment variables
or an IAM role. A Bedrock bearer token alone cannot authenticate S3 Vectors.
Temporary credentials require AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY and
AWS_SESSION_TOKEN. Never commit their values.

The AWS identity needs s3vectors:GetIndex, PutVectors, QueryVectors and
GetVectors on the index, plus bedrock:InvokeModel for the chosen embedding/chat
models. Cross-account indexes also need the appropriate resource policy.
An index encrypted with a customer KMS key needs corresponding KMS access.

## Build and run (PowerShell, repo root)

```powershell
docker build -t clearcall-backend .
# Stop the previous container if one exists:
docker stop clearcall-backend
$envPath = (Resolve-Path '.\langgraph-backend\.env').Path
docker run --rm --name clearcall-backend -p 2024:2024 -p 8001:8001 --env-file "$envPath" --mount "type=bind,source=$envPath,target=/app/.env,readonly" clearcall-backend
```

## Ingest (second terminal)

```powershell
Invoke-RestMethod -Method Post -Uri 'http://localhost:8001/ingest?target=s3vectors&dry_run=true' | ConvertTo-Json -Depth 10
$upload = Invoke-RestMethod -Method Post -Uri 'http://localhost:8001/ingest?target=s3vectors'
$upload | ConvertTo-Json -Depth 10
```

Only after a successful response with vectors_uploaded equal to chunks, copy
corpus_version into S3_VECTORS_CORPUS_VERSION in the environment file. Recreate
the container using the stop/run commands above (no rebuild needed).
Each process reads configuration at startup. Ingestion does not automatically
switch the active version. Do not activate a failed or partial upload.

## Test retrieval directly

```powershell
$body = @{ question = 'What fees must be disclosed to the client?'; top_k = 3 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri 'http://localhost:8001/search' -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 10
```

Expect backend=s3vectors and passages with text, source, page, distance,
vector_key and the configured corpus_version. Pages in `/search` remain
zero-based; the pipeline converts citations to one-based pages. Distance is
cosine distance (lower is closer), not a compliance confidence score.
This endpoint uses embeddings and S3 retrieval, without calling the chat model.

Then send the normal request to POST http://localhost:2024/runs/wait.
Guidance currently runs for targeted/exhaustive items only. Use one of those
items to test integration; deterministic-only checklists do not exercise it.
The current branch deliberately disables applicability/consistency nodes;
this integration does not re-enable them.

S3 retrieval failures propagate instead of silently using FAISS or returning
no guidance. Read `docker logs --tail 100 clearcall-backend` for the cause.
Missing corpus version, wrong dimensions, no matching vectors or invalid
citation metadata are errors. Set REGULATION_SEARCH_BACKEND=faiss and recreate
the container to explicitly return to local retrieval.

Offline tests (with backend packages installed):
```powershell
python -m unittest discover -s langchain-backend/tests -v
python -m unittest discover -s langgraph-backend/tests -v
```

Known baseline: two existing full-graph tests still expect the disabled
applicability/consistency workflow and fail on this branch. S3 ingestion,
retrieval, API, and guidance tests pass independently. Live AWS validation
requires the account owner's credentials and index access.
