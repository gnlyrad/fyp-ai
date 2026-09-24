# ClearCall compliance pipeline

The graph evaluates a transcript against a checklist:

```text
ingestion -> classification -> applicability -> guidance -> retrieval
          -> analysis (per question) -> consistency -> output
```

## Run with Docker (PowerShell, repository root)

```powershell
docker build -t clearcall-backend .
$envPath = (Resolve-Path ".\langgraph-backend\.env").Path
docker run --rm --name clearcall-backend -p 2024:2024 -p 8001:8001 --env-file "$envPath" --mount "type=bind,source=$envPath,target=/app/.env,readonly" clearcall-backend
```

Stop the previous container before starting its replacement:
`docker stop clearcall-backend`. Rebuild after source edits.

The environment file needs valid Bedrock credentials, for example
`AWS_BEARER_TOKEN_BEDROCK`. Optional LangSmith tracing uses
`LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and `LANGSMITH_PROJECT`.
The startup script launches LangGraph on port 2024 and the regulatory RAG
service on port 8001. Keep the terminal open.

For a Python environment, install from the repo root with
`python -m pip install -r langgraph-backend/requirements.txt`,
then run `langgraph dev` from this directory. The LangGraph configuration
expects a repository-root .env file for that path.

## Postman request

Send `POST http://localhost:2024/runs/wait`, Body -> raw -> JSON,
with Content-Type application/json. Supply the Transcribe file's results object,
not jobName/accountId/status. This abbreviated example illustrates the shape;
use your actual transcript and timing values:

```json
{
  "assistant_id": "compliance",
  "input": {
    "results": {
      "transcripts": [{"transcript": "Hello."}],
      "speaker_labels": {"speakers": 1, "segments": []},
      "items": [],
      "audio_segments": [
        {
          "id": 0,
          "transcript": "Hello.",
          "speaker_label": "spk_0",
          "start_time": "0.2",
          "end_time": "1.1",
          "items": []
        }
      ]
    },
    "checklist": [
      {
        "id": "greeting",
        "question": "Did the RM greet the client?"
      }
    ]
  }
}
```

When the speakers' roles have been confirmed, add a speaker_roles object next
to results inside input, for example `{"spk_0": "RM", "spk_1": "Customer"}`.
This is an example, not an assumption about any supplied recording. The backend
never determines a person's role from the order in which they speak. Unmapped
speakers stay neutral; role-dependent checks return Unclear until roles are
supplied. Caller-provided roles have speaker_role_source=caller. Plain text
without speaker metadata remains accepted but cannot establish role attribution.

Input precedence is results > audio_segments > transcript. The legacy
input.audio_segments and input.transcript formats remain accepted.
Within results, audio_segments takes priority over transcripts[0].transcript.
The text-only fallback does not invent speaker labels or times.

Optional checklist items can specify question_type, assessment
(presence/prohibition/quality), pass_condition, and subject_role (RM/Customer/any).
Items with missing or invalid classification metadata are classified by the model.
Generic pass conditions (e.g. `pass` or `all requirements are met`) are rejected;
only invalid/missing entries are retried once. Unresolved entries carry
`classification_status: needs_review` and block review even for an N/A finding.
This validation rejects obvious placeholders; it does not prove semantic correctness.
Missing/placeholder evidence queries receive a question-derived search phrase,
with `evidence_query_source` recording `classifier`, `provided`, or `question_fallback`.
Known
pressure-selling and return-guarantee questions always use exhaustive coverage.
Omit checklist to use the built-in checklist.

Save the API **response** as a JSON file. Creating a blank local file does not
save a response. The backend does not automatically create output_vN.json.

## Returned state and evidence

The response has exactly five top-level keys — `transcript`, `chunks`,
`checklist`, `findings`, `summary` — with no `results`, `audio_segments`,
`report`, `applicability`, `consistency` or `retrieval_diagnostics` in the
payload. Shared data is stated once; findings reference other sections by ID
instead of re-embedding them.

- `transcript` is the reconstructed full call text. Character offsets used
  elsewhere (`spans[].start`/`end`) index into this string.
- `chunks` lists each chunk once: `chunk_id`, `speaker` (mapped from
  `speaker_label`, omitted for plain-text input with no diarization), `text`,
  `start_seconds`, `end_seconds`. `source_segment_id` and `word_timings` are
  not returned. Word timings still exist internally and still drive
  word-precision span timing during analysis — they are just no longer part
  of the payload.
- `checklist` carries question metadata once: `id`, `question`,
  `question_type`, `pass_condition`. Findings key back to this via
  `checklist_id` instead of repeating it.
- `findings` has one entry per checklist item: `checklist_id`, `verdict`,
  `reason`, `evidence_chunk_ids` (the chunk IDs retrieval selected as
  evidence — the replacement for embedding full evidence-chunk objects),
  `evidence_status`, `requires_review`, `regulatory_basis`, and `spans`.
  - `evidence_status` replaces the old automatic confidence field. Statuses
    include direct_evidence, absence_after_full_review, conflicting_evidence,
    insufficient_evidence, not_applicable, applicability_uncertain,
    speaker_role_unverified, classification_unavailable,
    coverage_incomplete, and analysis_unavailable.
  - `requires_review` is retained alongside `evidence_status` because the two
    distinguish, for example, "Unclear because the applicability gate failed"
    from "Unclear because evidence was weak".
  - `regulatory_basis` is trimmed to `{source, page}` refs; the full MAS
    passage text is no longer included.
  - `spans[]` is trimmed to `text`, `start`, `end`, `label`, `chunk_id`,
    `start_seconds`, `end_seconds`. `start`/`end` are character offsets into
    `transcript` used for highlighting; `label` (compliant/violation/context)
    drives the highlight colour; `chunk_id` ties a span back to `chunks`.
    `timing_precision` is no longer on returned spans (word-level timing is
    still computed internally, just not surfaced).
- `summary` is today's report summary, promoted to the top level, with
  `warnings` nested inside it (`summary.warnings`, e.g. *"Speaker identities
  are preserved; some RM/Customer roles are unverified."*). It carries
  `total`, `pass`, `fail`, `unclear`, `not_applicable`, `compliant_spans`,
  `violation_spans`, `context_spans`, `requires_review`, `review_reasons`,
  `evidence_status_counts`, `by_type`, and `regulatory_sources`.

`applicability_node` and `consistency_node` are currently disabled, so
applicability decisions and cross-call consistency checks are not part of
findings or the response.

Example (abbreviated):

```json
{
  "transcript": "Hello, this is your RM calling about...",
  "chunks": [
    { "chunk_id": "chunk-00", "speaker": "spk_0", "text": "Hello, this is your RM calling about...",
      "start_seconds": 0.0, "end_seconds": 5.4 }
  ],
  "checklist": [
    { "id": "wm_q01", "question": "Did the RM disclose the fee structure?",
      "question_type": "deterministic", "pass_condition": "Fee structure is explicitly stated" }
  ],
  "findings": [
    { "checklist_id": "wm_q01", "verdict": "Pass", "reason": "The RM disclosed the fee structure.",
      "evidence_chunk_ids": ["chunk-02"],
      "evidence_status": "direct_evidence", "requires_review": false,
      "regulatory_basis": [{ "source": "SFA-Guidelines.pdf", "page": 15 }],
      "spans": [
        { "text": "our management fee is 1.5% annually", "start": 120, "end": 156,
          "label": "compliant", "chunk_id": "chunk-02", "start_seconds": 5.4, "end_seconds": 8.1 }
      ] }
  ],
  "summary": {
    "total": 1, "pass": 1, "fail": 0, "unclear": 0, "not_applicable": 0,
    "requires_review": false, "warnings": []
  }
}
```

An unrelated conversation is useful for testing N/A handling, not for judging
the original investment call. Diarization is preserved from Transcribe; the
backend cannot correct incorrectly merged speakers without reviewed source data.

## Limits and configuration

Default region: ap-southeast-1.
Default judge/classifier: apac.amazon.nova-pro-v1:0 (CLEARCALL_MODEL_ID). Anthropic
profiles suit the strict structured-output schemas better, but need the Anthropic
use case details form submitted for the AWS account before they can be invoked.
Default embedding model: cohere.embed-english-v3 (BEDROCK_EMBEDDING_MODEL).

| Variable | Default | Meaning |
|---|---|---|
| CLEARCALL_MAX_REVIEW_CHARS | 60000 | Maximum context characters for a whole-call model review |
| CLEARCALL_CHUNK_SENTENCES | 3 | Plain-text sentences per chunk |
| CLEARCALL_TOP_K | 3 | Targeted retrieval candidates |
| CLEARCALL_DETERMINISTIC_TOP_K | 5 | Presence-check retrieval candidates |
| CLEARCALL_MIN_RELEVANCE | 0.25 | Minimum semantic cosine score |
| CLEARCALL_GUIDANCE_TOP_K | 3 | Regulatory passages per guided item |
| CLEARCALL_GUIDANCE_CACHE | 1 | Set to 0 to disable guidance caching |

Oversized calls return a review-needed result rather than silently truncating
text or claiming complete coverage. For longer calls, a future batching strategy
must retain complete coverage and cross-batch contradiction checks.

Embeddings failing triggers a visible lexical fallback. Lexical matching
normalises hyphens and common domain word forms (objectives/objective,
cancellation/cancel, etc.) while preserving the original quotes.

Regulatory passages still use Bedrock embeddings plus local FAISS.
POST http://localhost:8001/ingest builds that index;
POST http://localhost:8001/ask answers regulatory-corpus questions.
S3 Vectors remains an explicit ingestion destination in the other backend.

## Tests

From langgraph-backend:

```powershell
python -m unittest discover -s tests -v
```

Tests mock AWS/model calls and disable tracing. They cover ingestion and exact
state preservation, zero-based IDs, speaker-role checks, word timing, lexical
fallback, routing and polarity, applicability, omission coverage, independent
quote labels, duplicate evidence, consistency, and graph output.
