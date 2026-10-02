# Compliance call review pipeline

## What this project is

A proof of concept for UBS (SMU IS484 FYP) that reviews recorded client calls against a compliance checklist. It takes the S3 URL of an AWS Transcribe output, fetches it, runs a staged LangGraph workflow, and saves structured results with evidence for a compliance dashboard.

The checklist is general. The pipeline must not hard-code specific questions. Behaviour for each item comes from a config file (see "Checklist config").

The pipeline does not extract structured entities (orders, amounts, product names) from the transcript. It only refers back to the transcript. Product facts (what was discussed, what the advisor said about a product) are inferred from the transcript itself, judged against MAS rules pulled from RAG -- there is no product master. Likewise there is no client master or restricted-list table: identity, eligibility, and red flags are judged from what was said on the call.

## Tech stack

- Python 3.11+, LangGraph (Graph API: `StateGraph`, `Send`), LangChain for retrieval internals
- Gemini (via `langchain-google-genai`) for the LLM, structured output in `function_calling` mode (the default `json_schema` mode drops nested models such as evidence lists); fixed sampling seed (`COMPLIANCE_LLM_SEED`, default 42) because the model ignores temperature
- Regulatory rules come from the existing S3 Vectors MAS corpus via `langchain_backend.retrieval.RegulationSearch` -- no ChromaDB, no Titan embeddings
- AWS Transcribe (with speaker labels) as input, fetched from S3 by `ingest` (`s3.py`, boto3; presigned URLs via plain HTTPS); output is JSON for now (`ResultWriter` protocol), Postgres later
- Pydantic for all structured LLM output
- pytest for tests

## Rules for working in this repo

- All LLM calls go through `llm.get_llm_response()`. Never call the model directly from a node. This keeps a later model swap to one file.
- Every LLM call returns validated Pydantic output. If validation fails, retry once, then return a result with status `CANNOT_DETERMINE` and an error note. Never crash the graph. Transient provider errors (503/429/timeouts) are waited out separately with backoff (`COMPLIANCE_LLM_BACKOFF_S`, default `5,15,30,60`) before counting as a failure.
- Evidence is always a list of `{segment_id, start_char, end_char}` refs into segment text, resolved in code from short LLM quotes (`nodes/_common.py:resolve_evidence`); quotes are never stored, only the resolved refs.
- Prefer plain code over an LLM whenever the task is deterministic (formatting, lookups, rules, mapping).
- Parallel branches write to state only through `Annotated[list, operator.add]` fields.
- Keep prompts in `prompts/`, keep checklist content in `config/checklist.yaml`. No question text in node code.

## Pipeline stages

```
S3 URL of Transcribe JSON
  │
  ▼
ingest            (code)  fetch from S3, merge words + speaker labels, one segment per speaker turn, stable IDs
  │
  ▼
quality_check     (code)  flag short calls, low confidence, single speaker
  │
  ▼
role_inference    (LLM)   map spk_0 / spk_1 to Advisor / Client
  │                        skip if the recording is dual-channel
  ▼
STAGE 1: FACTS
  └─ call_intent  (LLM)   enquiry | execution_only | advisory
  │
  ▼
gate              (code)  decide which Stage 2 items apply (applies_when)
  │                        implemented, NOT wired in yet: all Stage 2 items run
  │
  ▼
STAGE 2: CHECKS (Send fan-out, one branch per applicable item)
  └─ check_item   (LLM)   rule lookup + answer with sub-criteria
  │
  ▼
verify            (LLM)   re-check every FAIL, red flag, and low-confidence result
  │
  ▼
STAGE 3: DECISIONS
  └─ decide       (code first, LLM only for unclear cases)
       escalation → post-call actions → task ownership
  │
  ▼
aggregate         (code)  fill missing branches, compute summary
  │
  ▼
output            (code)  write JSON (Postgres later)
```

## File structure

```
config/
  checklist.yaml        checklist items and their metadata
  ownership.yaml         action type -> owning role
  escalation_rules.yaml  deterministic escalation rules
data/
  fixtures/               synthetic Transcribe JSON calls
prompts/
  role_inference.py
  call_intent.py
  check_item.py
  verify.py
  decide.py
nodes/
  _common.py             format_transcript, resolve_evidence, evidence_text
  ingest.py
  quality_check.py
  role_inference.py
  call_intent.py
  gate.py
  check_item.py
  verify.py
  decide.py
  aggregate.py
  output.py
llm.py                  get_llm_response()
s3.py                   fetch the Transcribe JSON from an S3 URL
retrieval.py            cached rule lookup per checklist item
schemas.py              Pydantic models
state.py                ComplianceState
graph.py                graph wiring
main.py                 run on one S3 URL or Transcribe JSON file
```

## Checklist config

`config/checklist.yaml` is only the *default* checklist. A human can edit it
and send the result as the request's `input.checklist`; `config.py`'s
`resolve_checklist(raw)` turns that into the checklist for one run, `ingest`
stores it in state (`checklist`), and every node reads it back via
`config.active_checklist(state)`. `config.get_item(item_id, checklist=None)`
looks an id up in a given checklist, or the default when none is given.

Rules for `resolve_checklist`:

- `None` or `[]` -> the default checklist unchanged.
- Otherwise the caller's items replace the defaults exactly, in caller
  order. A supplied id never inherits YAML fields, even if it matches a
  default id.
- A new id needs only `id` and `question`; it becomes a Stage 2
  `criteria` check with `applies_when: {always: true}`, `rule_queries:
  [question]` (searched live -- see "Retrieval"), and one sub-criterion
  `{id: "met", text: pass_condition or question}`.
- Call intent, escalation, actions and ownership are separate structured
  outputs and are not checklist items.
- Unknown keys (e.g. an old-format `question_type`) are ignored.
- Raises `ValueError` for: a non-list checklist; a non-dict entry; an entry
  missing `id`; a new item without `question`; a duplicate id; a new item
  that isn't Stage 2 `criteria`; an entry that changes the `stage` or
  `answer_type` of a default item; an unknown `applies_when` key; a
  checklist with no items; and duplicate sub-criterion ids
  within one item.

Each item in `config/checklist.yaml` has this shape:

```yaml
- id: C05
  name: Client understanding and disclosures
  stage: 2                     # 1 facts, 2 checks, 3 decisions
  answer_type: criteria        # yes_no | category | criteria | entities | list
  applies_when:
    call_intent: [advisory, execution_only]
  depends_on: []
  rule_queries:                # used to build the cached rule lookup
    - "risk disclosure requirements investment products"
  question: >
    Were the important risks, costs, restrictions, potential losses,
    and conflicts explained, and did the client appear to understand them?
  sub_criteria:
    - {id: risks, text: "Key risks explained"}
    - {id: costs, text: "Costs and fees explained"}
    - {id: restrictions, text: "Restrictions explained"}
    - {id: losses, text: "Potential losses explained"}
    - {id: conflicts, text: "Conflicts of interest disclosed"}
    - {id: understanding, text: "Client appeared to understand"}
```

`applies_when` supports two condition keys: `always` and `call_intent: [...]`. A sub-criterion may set `absence: true` for "nothing indicates X" checks, which can PASS without evidence; every other PASS needs at least one evidence ref. A sub-criterion may also set `warn_if_unconfirmed: true` to get a `source: "rule"` reviewer warning whenever it ends up NOT_APPLICABLE or CANNOT_DETERMINE (see "Warnings" above).

Current items:

| ID | Name | Stage | Answer type | Applies when |
|---|---|---|---|---|
| C01 | Identity and authority | 2 | criteria | always |
| C03 | Instruction validity | 2 | criteria | call_intent in [execution_only, advisory] |
| C04 | Product eligibility | 2 | criteria | call_intent in [advisory, execution_only] |
| C05 | Client understanding and disclosures | 2 | criteria | call_intent in [advisory, execution_only] |
| C06 | Restrictions and red flags | 2 | list | always |

There is no order/entity extraction item: the pipeline never extracts structured entities. C04's product facts come from what the advisor said about the product on the call, not a product master.

The prompt for a check only sees `question`, `sub_criteria`, the relevant rules, the transcript segments, and the call intent. All other fields are for the pipeline.

## Answer statuses

Use one enum everywhere:

- `PASS`
- `FAIL`
- `PARTIAL` (some sub-criteria pass, some fail)
- `NOT_APPLICABLE` (gated out)
- `CANNOT_DETERMINE` (transcript is silent, data is missing, or the LLM output failed)

`CANNOT_DETERMINE` is a valid, honest result. Never force a guess.

### Warnings

Every result goes to human evaluation, so `check_item` and `verify` also
attach `ReviewWarning`s -- possible or minor issues a reviewer should look
at even when nothing failed. A warning **never** changes any status or the
rollup; it is an extra signal only. Sources:

- `source: "llm"` -- the check_item LLM call reports anything it didn't
  mark FAIL but thinks a reviewer should see (ambiguity, something that may
  have happened off-call, a NOT_APPLICABLE/CANNOT_DETERMINE a reviewer
  could reasonably disagree with).
- `source: "rule"` -- code adds one whenever a sub-criterion configured
  with `warn_if_unconfirmed: true` ends up NOT_APPLICABLE or
  CANNOT_DETERMINE, and again (low severity) whenever `verify` downgrades a
  FAIL to CANNOT_DETERMINE, so the reviewer still sees the original finding.

Over-reporting is intentional: since a human reviews every call, it is
safer to surface a possible issue than to omit it.

## Schemas (in `schemas.py`)

Evidence is a resolved ref, never quoted text:

```python
class EvidenceRef(BaseModel):
    segment_id: str
    start_char: int          # relative to that segment's own `text`
    end_char: int
    verified: bool = False

class EvidenceQuote(BaseModel):   # LLM-output only, never persisted
    segment_id: str
    quote: str = ""
```

```python
class SubCriterionResult(BaseModel):
    id: str
    status: Status
    evidence: list[EvidenceRef]
    note: str

class ItemResult(BaseModel):
    item_id: str
    status: Status
    confidence: float            # 0 to 1
    sub_criteria: list[SubCriterionResult]
    evidence: list[EvidenceRef]
    rule_refs: list[str]         # IDs of rule chunks relied on
    reasoning: str               # short, for reviewers
    verified: bool = False
    warnings: list[ReviewWarning]  # possible/minor issues for a human reviewer; never affect status

class RedFlag(BaseModel):
    category: Literal["aml", "sanctions", "fraud", "market_abuse",
                      "restricted_product", "suspicious_instruction", "other"]
    severity: Literal["low", "medium", "high"]
    description: str
    evidence: list[EvidenceRef]

class Action(BaseModel):
    action_type: str              # key in ownership.yaml
    description: str
    owner_role: str | None        # filled by code
    evidence: list[EvidenceRef]
```

No `Order`/entity schema. LLM-output models (`IntentObservation`, `ProposedAction`) cite `evidence: list[EvidenceQuote]`; `nodes/_common.py:resolve_evidence` turns those into persisted `EvidenceRef`s.

## State (in `state.py`)

```python
class ComplianceState(TypedDict):
    call_id: str
    s3_url: str                            # S3 location of the Transcribe JSON (input)
    raw_transcript: dict
    results: dict                          # AWS Transcribe `results` object (raw input)
    checklist: list[dict]                  # this run's checklist (ChecklistItem dumps), resolved by ingest
    transcript: str                        # formatted transcript, one line per turn
    segments: list[dict]                   # {id, speaker, role, start, end, text, ...}
    quality_flags: list[str]
    call_intent: dict | None               # {category, confidence, evidence, reached, reasoning}
    applicable_items: list[str]
    item_results: Annotated[list[ItemResult], operator.add]
    final_results: list[ItemResult]        # aggregate's deduplicated, gap-filled output
    verified_results: list[ItemResult]
    red_flags: Annotated[list[RedFlag], operator.add]
    verified_red_flags: list[RedFlag]
    actions: list[Action]
    escalation: dict | None                # {required, reasons, hold_transaction, source, rule_ids}
    errors: Annotated[list[str], operator.add]
```

No `orders` / `product_context`: there is no order extraction and no product master.

## Node details

**ingest:** If `s3_url` is given, fetch the Transcribe JSON from S3 first (`s3.py:fetch_transcribe_json`; accepts `s3://`, S3 HTTPS, and presigned URLs; boto3 uses the standard AWS credential chain, a presigned URL needs none). Input precedence: `s3_url` > `results` > `raw_transcript`; a failed fetch raises `ValueError`. Then join Transcribe `items` with `speaker_labels.segments` by timestamp. One segment per speaker turn (a maximal run of consecutive speech by the same speaker), stable ID (`seg_0001`, and so on). Segment `text` is immutable after this point -- evidence offsets downstream are always relative to it.

**quality_check:** Add flags for: call under a set length, average word confidence below a threshold, only one speaker. Flags do not stop the graph but are shown on the dashboard and lower confidence.

**role_inference:** One cheap LLM call on the first few segments. If the recording is dual-channel, map channels to roles in code and skip the LLM.

**call_intent:** Classify as `enquiry`, `execution_only`, or `advisory`. A call can move from enquiry to order, so classify the most regulated intent reached in the call. The LLM cites short verbatim quotes per observation; code resolves them to evidence refs and keeps only observations with at least one resolved ref.

**gate:** Read `applies_when` for each Stage 2 item and set `applicable_items`. Write `NOT_APPLICABLE` results for the rest, citing the `call_intent` evidence. If call intent is undetermined, intent-gated items still apply. Implemented in `nodes/gate.py` but not wired into `graph.py` yet; the fan-out from `call_intent` sends every Stage 2 item until it is.

**check_item:** One branch per applicable item via `Send`. Loads the cached rules for that item from `retrieval.py`, builds the prompt from config, and returns an `ItemResult`. C06 also returns `RedFlag` objects. The LLM cites short quotes; code resolves them to `EvidenceRef`s via `resolve_evidence`.

**verify:** Runs once after all `check_item` branches join. Code first: drop any evidence ref whose segment id or span no longer checks out; a non-absence PASS left with no evidence becomes `CANNOT_DETERMINE`. Then, for Stage 2 items only (skipping `NOT_APPLICABLE`), one LLM call per item that has something to re-check -- every `FAIL` sub-criterion, every `PASS` sub-criterion when the item's confidence is under 0.7, and every red flag attached to the item (matched to the `answer_type: list` item) -- asking a second, skeptical reviewer whether the cited evidence actually supports each finding. Calls run concurrently (`ThreadPoolExecutor`, `MAX_WORKERS=4`). It can only uphold or downgrade: a `not_supported` sub-criterion becomes `CANNOT_DETERMINE` (never PASS, never FAIL); a `not_supported` red flag is dropped, and the sub-criterion it forced to FAIL is downgraded too unless another upheld flag of the same category remains. Item status and evidence are recomputed with the same `rollup_status` helper `check_item` uses. Never raises.

**decide:**
1. Escalation: apply `escalation_rules.yaml` rules-first, in order; every firing rule contributes (escalate/hold ORed, reasons/ids kept). An `item`+`status` rule fires on FAIL **or** PARTIAL (any failed sub-criterion is worth a human look, since every call is reviewed and over-reporting is preferred). There is also an `other_item_status` rule (`other_item_fail`, `when: {other_item_status: [FAIL, PARTIAL]}`): it fires for any Stage 2 item that has no `item`-specific rule of its own -- typically a check added through a caller-supplied checklist -- excluding `answer_type: list` items (C06's red flags are already covered by the `red_flag_severity` rules). This is what makes a failing caller-added check escalate. The LLM is used only as a fallback, when no rule fires at all but some Stage 2 result is `PARTIAL` or `CANNOT_DETERMINE` (an unclear case the rules don't cover) -- see `_has_unclear_stage2`. `Escalation.source` records which path decided it: `"rules"`, `"llm"`, `"none"` (nothing fired, nothing unclear), or `"undetermined"` (the LLM fallback itself failed).
2. Post-call actions: one LLM call proposes actions with evidence quotes; `action_type` is constrained via a dynamically-built Pydantic model (`_actions_output_model`) to exactly the keys in `ownership.yaml`, so an invalid action type fails validation instead of silently passing through. If escalation ended up required and the LLM didn't itself propose a `raise_case`, code adds one -- a case always gets raised when escalation is required.
3. Ownership: `action_type -> owner_role` is a plain lookup in `ownership.yaml`; the LLM never picks the owner.
4. The node returns `escalation` and `actions` directly. Each action already
   contains its `owner_role`; no C07/C08/C09 checklist results are created.

**aggregate:** Merge `item_results` with `verified_results` via `nodes/_common.py:merged_results` (verified wins per `item_id`) -- reused rather than duplicated. Every checklist item appears exactly once, in config order; a missing branch becomes `CANNOT_DETERMINE` "branch did not return". A result whose `item_id` isn't a real checklist item is dropped from `final_results` and reported in `errors` (guards a bad `Send` payload). Computes `summary`:

```python
summary = {
    "total": int, "counts": {status: int, ...all 5 keys...},
    "stage_counts": {1: {...5 keys...}, 2: {...}, 3: {...}},
    "quality_flags": [...], "call_intent": "advisory" | None,
    "escalation": {"required", "hold_transaction", "source", "rule_ids", "reasons"},
    "actions": int, "actions_by_owner": {role: int},
    "red_flags": int, "red_flags_by_severity": {severity: int},
    "warnings": int, "items_with_warnings": [item_id, ...],
    "verified_items": [item_id, ...], "errors": int,
    "needs_review": bool, "review_reasons": [str, ...],
}
```

`escalation` used to be the raw dict-or-None from `decide`; it is now always this shape (mirrors `Escalation.model_dump()`, plus `reasons`). `needs_review` is a triage signal for the dashboard only -- every call is reviewed by a human regardless -- true when escalation is required, any Stage 2 item is FAIL/PARTIAL/CANNOT_DETERMINE, any warning exists, any red flag exists, or a quality flag was raised; `review_reasons` gives short human-readable reasons for each.

**output:** Write the call, transcript, segments, results, red flags, actions, escalation, and summary as JSON (`ResultWriter` protocol). `JsonFileWriter` writes `<out_dir>/<call_id>.json`; `PostgresWriter` is a visible stub (`write()` raises `NotImplementedError("Postgres writer not implemented yet")`) so the interface exists without a real implementation, selected via `COMPLIANCE_WRITER=postgres`. The node never raises: a writer failure (including the Postgres stub) is caught and reported as an `errors` entry, not propagated.

The payload (`build_payload`, `schema_version: "1"`) adds `generated_at` (UTC ISO), `model: {id, structured_method, seed}` (from `llm.py`'s constants/env), and `rules: {rule_id: {text, source, page}}`: every `rule_id` referenced by any final result's `rule_refs`, collected once and referenced by id rather than repeated per item (per "Retrieval" below), via `retrieval.get_rule`. An id not found in the rule cache is omitted from `rules` and noted in `errors`. `render_report(payload)` renders a plain-text report (item table, escalation, actions, red flags, warnings, FAIL/PARTIAL sub-criteria with evidence text, error count) used by `main.py`; it never raises.

## Retrieval

Rules per item are fixed, so build the lookup once: run each item's `rule_queries` against the S3 Vectors MAS corpus, save the chunk IDs to a cache file, and load them at runtime. Rebuild the cache only when the rulebook or checklist changes. C06 uses one query per red flag category.

A caller-supplied checklist can add items or edit an item's `rule_queries`,
so those have no entry in the prebuilt cache. `retrieval.get_rules(item)`
uses the cache only when the item id is a default item *and* its
`rule_queries` still match the default's; otherwise it runs a live search
of the regulation corpus (`_live_rules`, `lru_cache`d per process by query
tuple) and registers the returned chunks in an in-memory index so
`get_rule(rule_id)` can still resolve them later (e.g. from `output.py`). A
live search can raise if the corpus is unreachable; `check_item` catches
that, judges the item without rules, and records an `errors` entry
(`check_item[<id>]: rule lookup failed, judged without rules: ...`) rather
than failing the branch.

## Tests

- `test_s3.py`: URL parsing, the boto3 and presigned paths, and `ingest` reading from `s3_url` (boto3/urllib stubbed, no network).
- Unit tests for `ingest`, `quality_check`, `role_inference`, `call_intent`, and `resolve_evidence` (fake LLM backend, no network -- `tests/compliance_pipeline/conftest.py` blocks live calls by default).
- `test_scaffold.py`: the graph compiles with the expected topology and runs end to end on stubs.
- `test_custom_checklist.py`: `config.resolve_checklist` replacement and validation rules; `retrieval.get_rules`' cache-hit vs. live-search branching; `check_item_node` running a caller-added item and recovering from a failed live rule lookup; and full graph checks for custom checklist ordering and generic escalation.
- Fixture calls in `data/fixtures/` covering the standard call shapes (enquiry, execution-only, advisory, a red flag, an unknown caller, and so on) once `check_item`/`decide` are implemented.
- LLM tests use a fake `get_llm_response()` that returns saved responses.

## Build order

1. `schemas.py`, `state.py`, `config/checklist.yaml` -- done
2. `ingest`, `quality_check`, `role_inference`, and their tests -- done
3. `call_intent` -- done
4. retrieval cache, `check_item` with `Send` -- done; `gate` is implemented (`nodes/gate.py`) but **not wired into `graph.py`** -- every Stage 2 item currently goes to `check_item` regardless of `applies_when`. To enable it, add the node between `call_intent` and the fan-out; `_route_to_checks` in `graph.py` already honours `applicable_items` when set.
5. `verify` -- done
6. `decide` with the rule files -- done
7. `aggregate`, `output`, `graph.py`, `main.py` -- done
8. Fixtures and scoring script -- in progress

## Open questions (check with UBS before relying on these)

- Are recordings dual-channel? If yes, drop role inference.
- Real field names for client classification, risk profile, and eligibility rules.
- Which escalation rules UBS actually uses.
- Whether labelled calls will be provided for scoring.
- Product master, client master, and restricted-list availability -- currently all inferred from the transcript rather than looked up.
