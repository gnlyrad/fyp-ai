# ClearCall compliance pipeline

The graph evaluates a transcript against a checklist:

```text
ingestion -> classification -> applicability -> guidance -> retrieval
          -> analysis (per question) -> consistency -> output
```

## Run with Docker (PowerShell, repository root)

```powershell
docker build -t clearcall-backend .
$envPath = (Resolve-Path ".\.env").Path
docker run --rm --name clearcall-backend -p 2024:2024 -p 8001:8001 --env-file "$envPath" --mount "type=bind,source=$envPath,target=/app/.env,readonly" clearcall-backend
```

Stop the previous container before starting its replacement:
`docker stop clearcall-backend`. Rebuild after source edits.

The repository-root `.env` needs `GOOGLE_API_KEY` (or `GEMINI_API_KEY`).
Chat defaults to `gemini-3.5-flash`; transcript and regulatory embeddings use
`gemini-embedding-001`. Both use the Gemini Developer API, never Bedrock.
Override models with `CLEARCALL_MODEL_ID` and `GOOGLE_EMBEDDING_MODEL`;
`CLEARCALL_TEMPERATURE` defaults to 1 for Gemini 3. Optional LangSmith tracing uses
`LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and `LANGSMITH_PROJECT`.
The startup script launches LangGraph on port 2024 and the regulatory RAG
service on port 8001. Keep the terminal open.

For a Python environment, change to `langgraph-backend` and install with
`python -m pip install -r requirements.txt`, then run `langgraph dev`
from that directory. The LangGraph configuration
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
The word `employee` is recognized as the RM subject. Explicit `subject_role`
metadata must be `RM`, `Customer`, or `any`; invalid values are rejected.
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

- results is preserved unchanged, including items and speaker_labels.
- audio_segments retains the original segments.
- transcript is reconstructed from the segment text.
- chunks starts at chunk-00 and follows supplied segment order; source_segment_id
  preserves the original segment id. Text chunking also starts at chunk-00.
- Each chunk carries source timing and speaker identity. Exactly aligned source
  items provide word_timings without changing transcript text or offsets.
- applicability records context-based applicable/not_applicable/uncertain
  decisions before compliance scoring. Missing disclosures in an applicable call
  do not make that call irrelevant.
- Applicability decisions use `applicability_coverage`. Findings expose this field
  alongside `analysis_coverage` (replacing the ambiguous finding `coverage` field).
  N/A findings report `analysis_coverage: skipped_not_applicable`, even when
  applicability reviewed the full transcript.
- retrieval_diagnostics records semantic, lexical_fallback, full_transcript or
  skipped retrieval, fallback reasons, selected chunk IDs and coverage.
- findings uses evidence_status instead of the old automatic confidence field.
  Statuses include direct_evidence, absence_after_full_review, conflicting_evidence,
  insufficient_evidence, not_applicable, applicability_uncertain,
  speaker_role_unverified, classification_unavailable, coverage_incomplete, and
  analysis_unavailable, and applicability_unavailable.
- Findings now separate `execution_status` (`success` or `prediction_failure`)
  from the semantic verdict. A prediction failure has **`verdict: null`**, an
  explicit `failure_category` and `failure_stage`, and requires review. Clients
  should display it as a prediction failure, not convert null back to Unclear.
  Completed uncertain analyses still return `Unclear`. The report summary and
  per-type breakdown expose `prediction_failure` separately from `unclear`.
  This is a change to the API's formerly always-string verdict field.
- `analysis_diagnostics` records each judge/audit invocation group: stage,
  attempt count, status, recovery, and error type/provider code for every failed
  attempt. Raw exception messages and model output are not copied into error
  diagnostics. `failure_error_type` and `failure_error_code` describe terminal
  judge/audit failures. The evaluator preserves these diagnostics in predictions
  and errors, and still accepts historical Unclear failure fallbacks.
- quote_verified proves only verbatim source text. A separate model review checks
  relevance and labels each passage compliant, violation or context. Labels are
  not forced by the finding's verdict. Duplicate spans within a finding are removed.
- Absence findings require whole-call coverage and have no fabricated quote or
  red/green highlight. Deterministic and exhaustive checks review the whole call;
  negative or uncertain targeted results receive one whole-call recheck after
  quote validation and auditing, including initially passing answers whose
  evidence was rejected. A completed full-call Unclear is never retried merely
  to obtain a binary verdict.
- spans carries timing_precision=word when exact source items align with the quote,
  otherwise segment or unavailable. Segment timing is not exact word timing.
- consistency records verified quote pairs for possible contradictions across
  the call, such as conflicting coupon conditions. These are review findings,
  not automatic legal conclusions.
- report includes summary, findings, spans, applicability, consistency and retrieval
  diagnostics, plus informational `warnings`. Any Fail/Unclear, unresolved
  classification or consistency check requires review. Unverified speaker roles
  block applicable findings that need attribution; they do not alone block N/A.
  Full-call coverage means text was supplied to the model, not
  a guarantee of perfect reasoning.

An unrelated conversation is useful for testing N/A handling, not for judging
the original investment call. Diarization is preserved from Transcribe; the
backend cannot correct incorrectly merged speakers without reviewed source data.

## Limits and configuration

Default region: ap-southeast-1.
Default judge/classifier: apac.amazon.nova-lite-v1:0 (CLEARCALL_MODEL_ID).
Default embedding model: gemini-embedding-001 (GOOGLE_EMBEDDING_MODEL).

| Variable | Default | Meaning |
|---|---|---|
| CLEARCALL_MAX_REVIEW_CHARS | 60000 | Maximum context characters for a whole-call model review |
| CLEARCALL_CONTEXT_MODE | routed | `routed` or `full_transcript`; changes transcript selection only |
| CLEARCALL_ANALYSIS_MAX_ATTEMPTS | 3 | Total application attempts per judge/audit stage, from 1 to 5 |
| CLEARCALL_CHUNK_SENTENCES | 3 | Plain-text sentences per chunk |
| CLEARCALL_TOP_K | 3 | Targeted retrieval candidates |
| CLEARCALL_DETERMINISTIC_TOP_K | 5 | Presence-check retrieval candidates |
| CLEARCALL_MIN_RELEVANCE | 0.25 | Minimum semantic cosine score |
| CLEARCALL_GUIDANCE_TOP_K | 3 | Regulatory passages per guided item |
| CLEARCALL_GUIDANCE_CACHE | 1 | Set to 0 to disable guidance caching |

Oversized calls return a review-needed result rather than silently truncating
text or claiming complete coverage. For longer calls, a future batching strategy
must retain complete coverage and cross-batch contradiction checks.

Each request may override the context/retry defaults with top-level input fields
`"context_mode": "full_transcript"` and `"analysis_max_attempts": 3`.
Full-transcript mode preserves each question's classification, prompt, and
regulatory guidance, while skipping transcript embedding and ranking. Regulatory
guidance may still need its own embedding calls. The context size limit applies
in both modes; no text is silently truncated.

The judge and evidence auditor retry transient connection, timeout, throttling,
and service errors using bounded exponential backoff with jitter. Malformed
structured output gets at most one repair with validation feedback, within the
same attempt budget. Missing/duplicate audit indices also trigger this repair.
Authentication, permission, configuration, and unexpected programming errors are
not retried by this recovery layer. If auditing alone fails, only the audit is
retried; unaudited spans are never published as accepted evidence. Setting
attempts to 1 disables these application retries. Provider SDK retries are
separate; diagnostics count application invocations, not HTTP requests.
Classification retains its existing two-attempt policy; applicability and
consistency do not use the new judge/audit retry budget.

Embeddings failing triggers a visible lexical fallback. Lexical matching
normalises hyphens and common domain word forms (objectives/objective,
cancellation/cancel, etc.) while preserving the original quotes.

Regulatory passages still use Gemini embeddings plus local FAISS.
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
