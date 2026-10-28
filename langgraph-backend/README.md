# ClearCall compliance pipeline

`compliance_pipeline/` reviews a recorded client call against a config-driven
checklist with a staged LangGraph workflow:

```text
ingest (fetches the Transcribe JSON from S3) -> quality_check -> role_inference
  -> call_intent -> check_item (one branch per item) -> verify -> decide
  -> aggregate -> output
```

See `CLAUDE.md` for the full design (pipeline stages, schemas, node details,
checklist config format). It uses Gemini for the LLM and writes JSON (a
Postgres writer is a stub, not yet implemented). It is registered as the
`compliance` graph in `langgraph.json`.

## Run with Docker (PowerShell, repository root)

```powershell
docker build -t clearcall-backend .
$envPath = (Resolve-Path ".\.env").Path
docker run --rm --name clearcall-backend -p 2024:2024 -p 8001:8001 --env-file "$envPath" --mount "type=bind,source=$envPath,target=/app/.env,readonly" clearcall-backend
```

Stop the previous container before starting its replacement:
`docker stop clearcall-backend`. Rebuild after source edits.

The startup script launches LangGraph on port 2024 and the regulatory RAG
service on port 8001. Keep the terminal open.

For a Python environment, install from the repo root with
`python -m pip install -r langgraph-backend/requirements.txt`,
then run `langgraph dev` from this directory. The LangGraph configuration
expects a repository-root .env file for that path.

## Input: an S3 URL to the Transcribe output

Send `POST http://localhost:2024/runs/wait` with the S3 location of the AWS
Transcribe output JSON (the file Transcribe writes, `{"jobName": ...,
"results": {...}}`). `ingest` downloads and parses it itself:

```json
{
  "assistant_id": "compliance",
  "input": {
    "s3_url": "s3://my-transcribe-output/calls/transcribe-job-123.json"
  }
}
```

Accepted `s3_url` shapes:

| Shape | Example | Needs AWS credentials? |
|---|---|---|
| S3 URI | `s3://bucket/key.json` | yes |
| Virtual-hosted HTTPS | `https://bucket.s3.ap-southeast-1.amazonaws.com/key.json` | yes |
| Path-style HTTPS (Transcribe's `TranscriptFileUri`) | `https://s3.ap-southeast-1.amazonaws.com/bucket/key.json` | yes |
| Presigned URL | `https://...amazonaws.com/key.json?X-Amz-Signature=...` | no, fetched with a plain GET |

The file may also be a bare `results` object. `call_id` defaults to the job's
`jobName`. Optional extra input keys: `call_id`, `speaker_roles`, `checklist`.
Input precedence is `s3_url` > `results` (an inline Transcribe `results`
object, still accepted) > `raw_transcript`. A failed fetch (missing object,
access denied, no credentials, invalid JSON) fails the run at `ingest` with a
readable `ValueError`.

### Optional: a custom `checklist`

`config/checklist.yaml` is only the default checklist. A human can edit it
and send the result as `input.checklist`, e.g.:

```json
{
  "assistant_id": "compliance",
  "input": {
    "s3_url": "s3://my-transcribe-output/calls/transcribe-job-123.json",
    "checklist": [
      {"id": "C01", "question": "Was the caller's identity verified using specific account details?"},
      {"id": "fees_disclosed", "question": "Were all fees disclosed?"}
    ]
  }
}
```

The caller's items replace the default checklist exactly, in the order given.
Every supplied item is an independent Stage 2 check; it does not inherit YAML
fields even when its id matches a default item. Call intent, escalation,
post-call actions and ownership are always computed as separate structured
outputs, not inserted as C02/C07/C08/C09 checklist results. See `CLAUDE.md`
"Checklist config" for the full rules and error cases.

A new or edited item's rules are looked up live in the MAS corpus for that
run (no prebuilt cache entry exists for it); if the regulation-search
backend isn't reachable, the check still runs, just without rule text, and
an `errors` entry records the lookup failure. A failing added check
escalates the call via the `other_item_fail` rule in
`escalation_rules.yaml`, the same way a failing default item escalates via
its own rule.

### AWS credentials for S3

Every shape except a presigned URL is read with boto3 `GetObject`, which uses
the standard AWS credential chain: `AWS_ACCESS_KEY_ID` +
`AWS_SECRET_ACCESS_KEY` (+ `AWS_SESSION_TOKEN` for temporary credentials) in
`.env`, `AWS_PROFILE` / `~/.aws/credentials`, or an attached IAM role when
deployed on AWS. The identity needs `s3:GetObject` on the object (and
`kms:Decrypt` if the bucket uses a customer-managed KMS key).
`AWS_BEARER_TOKEN_BEDROCK` only authorizes Bedrock and **cannot** read S3.
The region comes from the URL when it names one, else `AWS_REGION`
(default `ap-southeast-1`).

## Run from the command line

```powershell
python -m compliance_pipeline.main s3://my-transcribe-output/calls/job.json --out output
python -m compliance_pipeline.main ..\transcribe_73eee7cd_9408_4627_a726_a95fb725bb17_669c5fe4b7d1f44f.json
python scripts\run_request.py requests\s3_request.json output\recent_output.json
```

`main` takes an S3 URL, a local Transcribe job file, or a request file with an
`input` key. Add `--quiet` for a summary-only report.

## Configuration

Environment variables (all optional; `GOOGLE_API_KEY` is required to reach the
real Gemini model):

| Variable | Default | Meaning |
|---|---|---|
| GOOGLE_API_KEY | -- | Gemini API key (from the repo-root `.env`) |
| COMPLIANCE_MODEL_ID | CLEARCALL_MODEL_ID or gemini-3.5-flash | Optional pipeline model override |
| COMPLIANCE_LLM_SEED | 42 | Fixed sampling seed (the model ignores temperature); `none` disables it |
| COMPLIANCE_STRUCTURED_METHOD | function_calling | LangChain structured-output method (`json_schema` drops nested models) |
| COMPLIANCE_MAX_OUTPUT_TOKENS | 4096 | Max tokens per LLM call |
| COMPLIANCE_LLM_TIMEOUT_S | 90 | Per-call request timeout (seconds) |
| COMPLIANCE_LLM_BACKOFF_S | 5,15,30,60 | Waits (seconds) before retrying a transient Gemini error (503/429/timeout) |
| COMPLIANCE_MIN_CALL_SECONDS | 60 | Below this, `quality_check` flags a short call |
| COMPLIANCE_MIN_AVG_CONFIDENCE | 0.80 | Below this average word confidence, `quality_check` flags low confidence |
| COMPLIANCE_CH0_ROLE | Advisor | Channel-0 role for dual-channel recordings (skips `role_inference`) |
| COMPLIANCE_WRITER | json | `json` (`JsonFileWriter`) or `postgres` (stub, not implemented) |
| COMPLIANCE_OUTPUT_DIR | output | Directory `JsonFileWriter` writes `<call_id>.json` into |
| AWS_REGION | ap-southeast-1 | S3 region when the URL doesn't name one |
| COMPLIANCE_S3_MAX_BYTES | 52428800 | Largest Transcribe JSON `ingest` will download |
| COMPLIANCE_S3_TIMEOUT_S | 30 | Timeout for presigned-URL downloads (seconds) |

Rebuild the cached regulatory rule lookup (only needed when the rulebook or
`config/checklist.yaml` changes):

```powershell
python scripts\build_rule_cache.py
```

## Tests

From `langgraph-backend`:

```powershell
python -m pytest tests/compliance_pipeline -q
```

Tests never reach Gemini or AWS: a fake LLM backend is installed by
`tests/compliance_pipeline/conftest.py`, and `test_s3.py` stubs boto3/urllib.


### Gemini integration after merging main

Chat and live regulatory embeddings use the shared Gemini Developer API client.
Set `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in `fyp-ai/.env`; Vertex AI and Bedrock
are not used for model calls. `CLEARCALL_MODEL_ID`, `CLEARCALL_TEMPERATURE`, and
`GOOGLE_EMBEDDING_MODEL` configure the shared clients. The pipeline also honors
`COMPLIANCE_MODEL_ID`, output-token/timeout limits, and `COMPLIANCE_LLM_SEED`.

Regulatory search defaults to local FAISS. Set `REGULATION_SEARCH_BACKEND=s3vectors`
only with a corpus ingested using the same Gemini embedding model and dimension;
an existing Cohere index cannot be reused. AWS credentials are needed only for
optional S3 input/storage/retrieval and deployment.

Results retain the native status and include `execution_status`, `failure_category`,
and `failure_stage`. Failed checks/verification are distinct from successful
`CANNOT_DETERMINE` answers. Provider diagnostics omit raw exception messages.

### Multilingual NLP reliability

The existing `compliance` graph continues to accept transcript JSON and a caller
checklist. Its public input/output shapes, model defaults, routing and verification
rollup are unchanged. `requirement_type` is still optional. No transcription or
audio processing changes are needed for these NLP improvements.

Evidence lookup accepts cosmetic width/case differences and whitespace next to
Chinese characters while retaining offsets into the original, unchanged text.
Negation, numbers, financial symbols and numeric punctuation are not replaced.
Translated or invented evidence still fails validation. A provided evidence quote
must contain original words; a legitimate absence finding uses its existing
absence-review representation rather than an empty quote.

Standard and native grouped responses retain independently parsed, unique item
answers if a sibling has a malformed result. The existing production recovery
process receives only the missing/unusable rows. Duplicate IDs are excluded;
missing statuses are formatting failures rather than implicit semantic unclear
answers. A completed valid PASS, FAIL or CANNOT_DETERMINE is not retried to select
a better-scoring answer. Existing attempt limits and native job receipt rules
still apply. Verification also validates any supplied counter-evidence before
using it; a valid semantic `not_supported` verdict keeps the existing conservative
downgrade behavior.

Checking and verification prompts explicitly cover English, Singlish, Mandarin,
code-switching, scoped consent, negation, withdrawals and corrected final orders.
Quotes remain in their original language. No extra model stage or blanket second
review is added. Missing/all-zero timing metadata yields the informational
`timing_unavailable` flag, rather than claiming a short recording. Known short
recordings still receive `short_call`.

The new `advisory_execution_multilingual_v1_development` dataset contains 175
development transcripts, 88 containing Chinese characters. Its manifest declares
placeholder timings and pending dialogue verification; no recording files are
included in that dataset. Treat it as a development benchmark, preserve the
separate 75-call held-out split, and do not claim audio/transcription accuracy
from those JSON files. The production code never reads this dataset or its labels.

For validation, freeze the model, grouping, reasoning, retrieval and retry settings
and compare the baseline and candidate on identical approved inputs. Report binary
decision coverage and detected breaches divided by **all** gold breaches, including
unclear/failed predictions in the latter denominator. Also report false clearances,
precision, per-language counts, reported model-token cost and incomplete usage.
Offline regressions establish recovery/evidence behavior, not improved benchmark
percentages. Live inference and evaluation remain a separately authorized step;
the evaluator, datasets, labels and `.env` are not modified by these changes.
