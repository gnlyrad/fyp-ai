# Offline NLP evaluation engine

This directory evaluates the committed, error-free synthetic transcripts against
the repository's actual compiled LangGraph pipeline. The runner invokes the graph
once per complete call, then scores all checklist findings returned by that
invocation. It does not regenerate data, simulate NLP logic, or call STT. The
same versioned artifacts are consumed by the local/shared evaluation dashboard.

## Dataset and graph input

`datasets/pilot/` is an immutable published snapshot from `fyp-be`. The loader
validates every manifest hash, checklist/ground-truth relationship, transcript
turn reference, and speaker-role mapping before execution. A read or validation
failure stops setup; the evaluator never substitutes another dataset.

The graph receives the existing adapter output:

```python
{
    "results": transcript_json,
    "speaker_roles": speaker_roles_json,
    "checklist": [{"id": item["id"], "question": item["question"]} for item in checklist_json],
}
```

Ground truth and evaluation-only checklist metadata never enter the graph.
Supplied checklist IDs are preserved exactly and all pilot questions run as
generic Stage 2 checks. Call intent, escalation, actions and ownership are
separate structured outputs, not extra checklist results. The current graph does
not wire its applicability gate, so every supplied checklist question runs. An
N/A prediction remains an unexpected outcome.

Validate the published dataset without invoking the graph:

```powershell
python -B evaluation/scripts/prepare_run.py --all --check-only
```

## Run the evaluator

From `fyp-ai/`, evaluate the full snapshot:

```powershell
python -B evaluation/scripts/run_evaluation.py --all
```

The staged graph uses the full transcript:

```powershell
python -B evaluation/scripts/run_evaluation.py --all --context-mode full_transcript --max-attempts 3
```

`--max-attempts` caps each LLM stage's total application attempts (default 3;
1 disables application retries). SDK retries are disabled. The old routed mode
is rejected for live runs. The resolved options and pipeline format are recorded
in `metadata.json`. The published dataset remains unchanged.

A live run needs a valid Gemini key, backend dependencies, and working regulatory
retrieval. Check execution errors and evidence quality alongside accuracy.

Run one or more selected calls:

```powershell
python -B evaluation/scripts/run_evaluation.py --call CALL_001 --call CALL_005
```

The real graph uses Gemini chat and embeddings. Set `GOOGLE_API_KEY` in
`fyp-ai/.env` and install the LangGraph backend dependencies. Network access
is required. Defaults: `gemini-3.5-flash`, `gemini-embedding-001`, temperature 1.
The regulatory FAISS index is namespaced by embedding model; its first query
builds a new Gemini index without overwriting the old Bedrock index.
Graph errors are persisted and do not become model abstentions.

The staged adapter maps PASS/FAIL/CANNOT_DETERMINE/NOT_APPLICABLE to the
legacy scorer vocabulary; PARTIAL maps to Fail. The original status is retained
in `graph_metadata.finding.native_status`. Evidence character offsets are mapped
back to published turn IDs only when ingested text and speaker sequences match.
References that cannot be mapped are recorded as invalid, not silently discarded.
Saved legacy runs can still be rescored through the original adapter.

Known permanent Gemini errors (401/403/404) stop retries immediately. Failed or
incomplete checks and failed verification are prediction failures, not semantic
abstentions. Other graph diagnostics appear in `errors.jsonl` and make the run
`completed_with_errors`. Historical results from the previous architecture are
not validation of the merged pipeline; run a fresh evaluation for that comparison.

## Evaluation families

Every checklist question is scored by exactly one family, saved as
`evaluation_family` in predictions, scores, `results.csv` and errors. It is
derived from the checklist `type` (`COMPLIANCE` is `compliance`; `CLIENT_KYC` and
`TRANSACTION` are `extraction`) and must agree with the ground-truth
`expected.kind`; the loader and scorer reject a mismatch. Question IDs are never used.
The checklist type stays as the source category (`question_type`).

Three dimensions are kept apart and never merged into one enum:

- `execution_status`: `success` or `prediction_failure`. `failure_category` says why
  (`analysis_unavailable`, `missing_prediction`, ...).
- `native_verdict`: the raw graph value (`Pass`/`Fail`/`Unclear`/`N/A`, or null for
  an operational failure in the repaired graph). Diagnostic only.
  A native `Unclear` is a compliance abstention only when execution succeeded; after a
  failure it is a fallback; for extraction it is ordinary output.
- The family-specific result below.

Legacy repaired graph failures have a null verdict and explicit `execution_status` and
`failure_category`. The adapter recognizes these before validating the verdict,
and continues recognizing historical `Unclear / analysis_unavailable` fallbacks.
Judge/audit `analysis_diagnostics`, failure stage, error type and provider error
code are retained under `graph_metadata.finding` in predictions and in failure
details in `errors.jsonl`. A null native verdict is grouped under `MISSING` in
the existing raw-verdict diagnostic histogram; consult execution status to
distinguish an explicit failed finding from a missing finding.

### Compliance

Gold is `PASS`/`FAIL`; `FAIL` is the positive class. Every compliance question has one
`compliance.outcome_status`: `binary_decision`, `abstention`, `prediction_failure` or
`unexpected_not_applicable`. They sum to the compliance question count.

- Decision coverage, abstention rate and prediction failure rate share the compliance
  question count as denominator.
- Binary accuracy, breach precision, breach recall and TP/TN/FP/FN use completed binary
  decisions only. A ratio with a zero denominator is `null`, never `0`.
- Breach outcomes decompose every gold `FAIL` into `detected`, `false_clearance`,
  `abstained`, `prediction_failure` or `unexpected_not_applicable`, with counts and rates.

### Structured extraction

Extraction is scored on one question: does the final selected evidence contain the
expected value? It has no PASS/FAIL semantics, so it has no accuracy, breach or
abstention metrics.

- `gold_value_status` (dataset): `PRESENT` or `NOT_STATED` (`expected_value` is null).
- `evaluator_support` (PRESENT rows): `SUPPORTED` (text, `dob_dd_month_yyyy`) or
  `UNSUPPORTED` (no deterministic comparison exists; zero in the pilot).
- `evidence_status`: `FOUND` or `NOT_FOUND`, only for a successful, PRESENT, SUPPORTED row.
- Headline **extraction evidence coverage** = FOUND / (PRESENT and SUPPORTED rows).
  A prediction failure on such a row stays in the denominator and is never `NOT_FOUND`.
  `NOT_STATED` and `UNSUPPORTED` rows are outside it. **Extraction prediction failure
  rate** = extraction prediction failures / all extraction questions.
- Text matching is token-boundary aware (`100` does not match `1000`); dates are
  normalized as calendar dates. There is no fuzzy matching and no LLM judge.
  Coverage is not value accuracy: the graph returns no dedicated extracted value.

### Evidence

Evidence metrics use final finding `spans` only, mapped through the graph's returned
chunks and their `source_segment_id`. Candidate retrieval chunks and
`reviewed_chunk_ids` are not predictions. Precision, recall and exact-set match are
aggregated per family (`compliance.evidence`, `extraction.evidence`), each with
`non_empty_gold` and `empty_gold` breakdowns. For extraction they are secondary
diagnostics. The evidence empty-set behavior is a project-specific convention:

| Gold set | Predicted set | Precision | Recall | Exact set match |
| --- | --- | ---: | ---: | ---: |
| Non-empty | Non-empty | Set formula | Set formula | Set equality |
| Non-empty | Empty | 0 | 0 | 0 |
| Empty | Non-empty | 0 | 0 | 0 |
| Empty | Empty | 1 | 1 | 1 |

Headline evidence results are macro means of per-question scores. A prediction failure
gets a zero penalty rather than empty/empty credit. Invalid evidence references stay in
the precision denominator, block exact match, and are reported separately; they are
never `UNSUPPORTED`. `evidence_all_questions` is a blended, diagnostic-only aggregate.

## Run artifacts

Each run creates a new `runs/<run_id>/` directory and refuses to overwrite an
existing ID:

```text
metadata.json
raw/<call_id>.json
predictions.jsonl
scores.jsonl
summary.json
results.csv
errors.jsonl
```

`summary.json` has `overall`, `by_call_id` and `by_question_id` nodes with the same
shape: `counts` (including `native_unclear_reconciliation`), `compliance`,
`extraction` and `evidence_all_questions`. A family block is `null` when the node has
no question of that family. `scores.jsonl` has shared fields plus a `compliance` or an
`extraction` block (the other is `null`), so a UI needs no rescoring. `results.csv` is a
single file, one row per `(call_id, question_id)`; fields of the other family are blank.
Token usage, estimated cost, and trace IDs are marked not exposed when the current graph
state does not expose them.

`metadata.json` records `run_kind` (`graph_execution` or `rescored`),
`artifact_schema_version`, `scorer_version`, `metric_definitions_version`, immutable
dataset/checklist/metric-definition fingerprints, and graph Git lineage (commit,
author, timestamp and dirty-worktree flag).

### Rescoring without model calls

`raw/<call_id>.json` holds each call's graph output, so scoring can be redone
deterministically into a NEW run without LangGraph or Gemini. The source run is only
read, and the dataset fingerprint must match:

```powershell
python -B evaluation/scripts/rescore_run.py --source-run 20260925T000441Z-e8578f7d
```

The derived run has `run_kind: rescored` and `derived_from` (source run ID, its scorer,
metric-definitions and schema versions, dataset fingerprint, graph commit,
`graph_invoked: false`).

### Version 2.0.0 compatibility

`artifact_schema_version` and `metric_definitions_version` 2.0.0, and scorer
`fyp-nlp-evaluator-v2.0`, break the v1 layout. Compare only runs of the same schema
version, or rescore v1 runs into v2.

| v1 | v2 |
| --- | --- |
| `execution_status` `missing_prediction` / `malformed_prediction` | `prediction_failure` plus `failure_category` |
| `predicted_verdict` filled from a failed fallback or an extraction question | null; the raw value is `native_verdict` |
| `answer.*`; status `scored_binary` | `compliance.outcome_status`; `binary_decision` |
| `gold_verdict`, `gold_value`, `gold_value_type`, `gold_kind` | inside `compliance` / `extraction`; `evaluation_family` |
| `extraction_evidence_coverage.{status,found}` (found, not_found, unsupported, no_value_expected, prediction_failure) | `extraction.{gold_value_status, evaluator_support, evidence_status, evidence_coverage}` |
| `selected_evidence`, `graph_metadata` in scores.jsonl | only in predictions.jsonl |
| summary `compliance.binary_accuracy` etc. | `compliance.binary.accuracy`, `compliance.breach_outcomes.*` |
| summary `compliance.native_unclear_outputs` (always 0: it compared `Unclear` to `UNCLEAR`) | `compliance.native_diagnostics.*`, corrected |
| results.csv `answer_status`, `extraction_coverage_status`, `extraction_expected_value_found`, `gold_value` | `compliance_outcome_status`, `extraction_evidence_status`, `extraction_evidence_coverage`, `expected_value` |

## Metric definitions

[`metric_definitions.json`](metric_definitions.json) is a versioned glossary for
dashboard tooltips. It is the single source of metric formulas, populations,
interpretations and status definitions. Each metric has a `summary_path`, relative to
one aggregate node of `summary.json`, so a dashboard reads values from the summary and
text from here. It is a shared metric reference, not a prediction or score file.

Its SHA-256 is recorded in every run and checked before two runs are compared, so do
not edit it for wording. Plain-English dashboard wording (labels, "what it tells you",
calculation, caveat, and human status labels) lives in
[`metric_presentation.json`](metric_presentation.json), keyed by the same metric and
status IDs. That file is UI-only and not fingerprinted; `tests/test_metric_presentation.py`
rejects entries for unknown IDs and evaluator vocabulary in sponsor-facing text.

## Evaluation dashboard

The dashboard is a React app (`web/`) over a small FastAPI API (`dashboard/api.py`).
The API reads through `EvaluationRepository`, so **Local** (files under `datasets/`
and `runs/`) and **Shared** (published history in Supabase) use one API contract. The
browser never reads run files or queries Supabase itself, and the dashboard never
recomputes a metric: values come from `summary.json`/Postgres, drill-down lists are
filters over score records (tested to reproduce the summary counts), and wording comes
from `metric_presentation.json`. Times are shown in SGT.

### First-time setup and everyday commands

Run everything from the `fyp-ai/` folder. Any Python 3.11+ works (the project `.venv` is optional).

| What you want | One-time setup | Command |
|---|---|---|
| **View the dashboard** | `python -m pip install -r evaluation\requirements.txt`, plus Node.js 20+ installed (used once, automatically, to build the web app) | `python evaluation\scripts\serve_dashboard.py`, then open http://127.0.0.1:8000 |
| **Publish a run** (to Shared) | the same install; put `SUPABASE_URL` and `SUPABASE_PUBLISHABLE_KEY` in the repo-root `.env` | `python -B evaluation\scripts\upload_run.py` |
| **Run an evaluation** (calls Gemini) | `cd langgraph-backend`, then `python -m pip install -r requirements.txt`, then `cd ..` (it must be run from inside that folder), and `GOOGLE_API_KEY` in `.env` | `python -B evaluation\scripts\run_evaluation.py --all` |
| **Run the tests** | `python -m pip install -r evaluation\requirements-dev.txt` (browser tests also need `python -m playwright install chromium`) | `python -m pytest evaluation/tests -q` |

Day to day you only need the command column. The first `serve_dashboard.py` run installs the web
app's packages and builds it; later runs rebuild only if the web source changed.

Restart it after pulling backend changes (the page and the server must match). Use `--default-source
local` to open on your local runs, `--port`, or `--no-build` to skip the automatic build. For
frontend development, run that server and `npm run dev` in `evaluation/web` (http://localhost:5173,
which proxies `/api` to port 8000). `web/dist` is a build output and is not committed.

The legacy Streamlit dashboard (`dashboard/app.py`) is kept temporarily as a parity reference and is
no longer installed by default: `python -m pip install -r evaluation\requirements-legacy.txt`, then
`python -m streamlit run evaluation\dashboard\app.py`.

Copy the project URL and publishable key from `.env.example` into the repository-root
`.env` file. Those are the only two values required to view Shared mode; there is no
dashboard login. Published synthetic evaluation rows are publicly readable through the
publishable key, and the same key can publish runs (insert and finalize only, never edit
or delete). No publisher account or password is needed. A service-role or secret key is
never required.

Publish a complete run. With no arguments the script lists your local runs (newest first,
marking which are ready, already published, or partial and local-only) and asks which to publish:

```powershell
python -B evaluation\scripts\upload_run.py
python -B evaluation\scripts\upload_run.py --run-id <RUN_ID>   # skip the menu
```

Progress is printed step by step. The command validates all required artifacts and full dataset
coverage before network access, then writes in batches (a run is a few dozen requests, not
one per row), and skips dataset work when the dataset snapshot is already published. Partial/debug
runs stay local. Repeating an identical publication is a no-op; reusing a run ID with different
contents is rejected; an interrupted publication resumes when re-run. A complete
`completed_with_errors` run remains publishable. Older runs without the lineage fingerprints must
first be rescored into a new immutable run:

```powershell
python -B evaluation\scripts\rescore_run.py --source-run <RUN_ID>
```

## Time, tokens and cost

Every run records its wall-clock time (per call and total) and, for new runs, chat-model token
usage and an estimated cost.

- **Tokens come from the pipeline itself.** The model-calling graph nodes are wrapped in
  `compliance_pipeline/usage.py`, which adds each node's token usage to a `usage` field of the
  graph state. The pipeline's output (and its written JSON payload) therefore includes
  `usage`, shaped `{model: {input_tokens, output_tokens, total_tokens, ...}}`, for any caller.
  Retried attempts are included. The evaluation reads that field (falling back to a callback
  around the call for graphs that do not report it). It works for any LangChain chat model,
  so trying another model needs no code change. Embedding calls are not counted.
- **Cost cannot come from the model.** Gemini (like most providers) returns token counts, not a
  price, so the pipeline reports tokens only and the evaluation turns tokens into money. A model's
  price is found in this order: (1) an entry you set in [`model_pricing.json`](model_pricing.json)
  (an optional override, e.g. a negotiated rate); (2) otherwise a public price catalog
  ([LiteLLM's](https://github.com/BerriAI/litellm), which cites each provider's own pricing page),
  downloaded at run time only when needed, with the Gemini Developer API entry preferred over
  resellers. Standard rates are used (not batch or priority) and thinking tokens are billed as
  output. If neither source has the model, or you are offline, tokens are still recorded and cost
  shows as unavailable, never zero. Trying another model needs no code or config change.
- **Prices are retained.** Each run copies the exact prices it used, with their source and date, into
  its own metadata (shown under Run details), so a model getting cheaper later never changes an
  old run's cost. `python -B evaluation\scripts\update_prices.py` refreshes the price book
  from the catalog and keeps the previous price in each entry's `history`. The catalog is
  community-maintained, so treat costs as estimates; pin a price in `model_pricing.json` to override it.
  Set `EVALUATION_PRICE_CATALOG_URL=off` to disable the lookup.
- Runs made before this was tracked show these values as empty, never zero. Everything is stored
  in `metadata.observability`, which Shared already stores, so no database change is needed.

Supabase setup and role provisioning are documented in
[`../supabase/README.md`](../supabase/README.md).

## Tests

Tests use synthetic fixtures and injected graph functions. They do not call
Gemini:

```powershell
python -m pytest evaluation/tests -q
```

Browser smoke test of the built dashboard (server running, Local source):

```powershell
.\.venv\Scripts\python.exe evaluation\tests\browser_web_dashboard.py http://127.0.0.1:8000
```
