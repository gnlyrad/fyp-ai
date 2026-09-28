# Offline NLP evaluation engine

This directory evaluates the committed, error-free synthetic transcripts against
the repository's actual compiled LangGraph pipeline. The runner invokes the graph
once per complete call, then scores all checklist findings returned by that
invocation. It does not regenerate data, simulate NLP logic, call STT, or build a
dashboard.

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
Applicability is deliberately bypassed for this evaluation by setting
`CLEARCALL_SKIP_APPLICABILITY=true` only during graph execution. All questions
are treated as applicable; an `N/A` output is normalized to
`NOT_APPLICABLE` and reported as unexpected.

Validate the published dataset without invoking the graph:

```powershell
python -B evaluation/scripts/prepare_run.py --all --check-only
```

## Run the evaluator

From `fyp-ai/`, evaluate the full snapshot:

```powershell
python -B evaluation/scripts/run_evaluation.py --all
```

Compare transcript selection while keeping the same model, question types,
prompts, guidance and retry budget:

```powershell
python -B evaluation/scripts/run_evaluation.py --all --context-mode routed --max-attempts 3
python -B evaluation/scripts/run_evaluation.py --all --context-mode full_transcript --max-attempts 3
```

Use `--max-attempts 1` to disable application-level judge/audit retries in either
condition. This does not revert the other workflow repairs, disable SDK retries,
or change classification's existing retry policy. Omitted options use
`CLEARCALL_CONTEXT_MODE` (default `routed`) and `CLEARCALL_ANALYSIS_MAX_ATTEMPTS`
(default `3`). Both resolved values are saved under `execution_configuration`
in `metadata.json`; each run remains immutable. These flags are applied to graph
input, without modifying the published dataset or sending gold metadata.

A faithful live run requires the complete backend dependencies, a valid Google API key, and working regulatory retrieval. Full-transcript mode skips only
transcript embeddings; guidance retrieval remains enabled as before. Compare
failure rate and decision coverage alongside accuracy, breach outcomes, evidence
quality and latency. A lower Unclear count alone is not proof of improvement.

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

Classification failures also preserve the provider error type and code in each
affected finding. Known permanent provider errors (for example,
Google HTTP `401`, `403` or `404`) stop classification
retries immediately; they require a credential/access fix, not prompt tuning.
Raw provider exception messages are not copied into classification findings.

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

New graph failures have a null verdict and explicit `execution_status` and
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
`artifact_schema_version`, `scorer_version`, `metric_definitions_version`, the dataset
fingerprint and the graph commit.

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

## Metric definitions for a future UI

[`metric_definitions.json`](metric_definitions.json) is a versioned glossary for
dashboard tooltips. It is the single source of metric formulas, populations,
interpretations and status definitions. Each metric has a `summary_path`, relative to
one aggregate node of `summary.json`, so a dashboard reads values from the summary and
text from here. It is a shared metric reference, not a prediction or score file.

## Tests

Tests use synthetic fixtures and injected graph functions. They do not call
Gemini:

```powershell
python -m pytest evaluation/tests -q
```
