# Compliance call review pipeline

## What this project is

A proof of concept for UBS (SMU IS484 FYP) that reviews recorded client calls against a compliance checklist. It takes an AWS Transcribe output, runs a staged LangGraph workflow, and saves structured results with evidence to PostgreSQL for a compliance dashboard.

The checklist is general. The pipeline must not hard-code specific questions. Behaviour for each item comes from a config file (see "Checklist config").

## Tech stack

- Python 3.11+, LangGraph (Graph API: `StateGraph`, `Send`), LangChain for retrieval internals
- Amazon Bedrock for the LLM and Titan Embeddings, ChromaDB for the rulebook vector store
- AWS Transcribe (with speaker labels) as input, PostgreSQL as output
- Pydantic for all structured LLM output
- pytest for tests

## Rules for working in this repo

- All LLM calls go through `llm.get_llm_response()`. Never call Bedrock directly from a node. This keeps a later swap to SageMaker (or another model) to one file.
- Every LLM call returns validated Pydantic output. If validation fails, retry once, then return a result with status `CANNOT_DETERMINE` and an error note. Never crash the graph.
- Evidence is always a list of transcript segment IDs (`evidence_ids`), never quoted text.
- Prefer plain code over an LLM whenever the task is deterministic (formatting, lookups, rules, mapping).
- Parallel branches write to state only through `Annotated[list, operator.add]` fields.
- Keep prompts in `prompts/`, keep checklist content in `config/checklist.yaml`. No question text in node code.

## Pipeline stages

```
Transcribe JSON
  │
  ▼
ingest            (code)  merge words + speaker labels, segment with stable IDs
  │
  ▼
quality_check     (code)  flag short calls, low confidence, single speaker
  │
  ▼
role_inference    (LLM)   map spk_0 / spk_1 to Advisor / Client
  │                        skip if the recording is dual-channel
  ▼
STAGE 1: FACTS (parallel)
  ├─ call_intent  (LLM)   enquiry | execution_only | advisory
  └─ extraction   (LLM + code)  orders and entities, then normalise
  │
  ▼
context_lookup    (code)  client record, product record, restricted lists
  │
  ▼
gate              (code)  decide which Stage 2 items apply (applies_when)
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
output            (code)  write to PostgreSQL
```

## File structure

```
config/
  checklist.yaml        checklist items and their metadata
  ownership.yaml        action type -> owning role
  escalation_rules.yaml deterministic escalation rules
data/
  mock/                 mock client, product, and restricted list tables (seed SQL or CSV)
  fixtures/             synthetic Transcribe JSON calls
prompts/
  role_inference.py
  call_intent.py
  extraction.py
  check_item.py
  verify.py
  decide.py
nodes/
  ingest.py
  quality_check.py
  role_inference.py
  call_intent.py
  extraction.py
  normalise.py
  context_lookup.py
  gate.py
  check_item.py
  verify.py
  decide.py
  aggregate.py
  output.py
llm.py                  get_llm_response()
retrieval.py            cached rule lookup per checklist item
schemas.py              Pydantic models
state.py                ComplianceState
graph.py                graph wiring
main.py                 run on one Transcribe JSON file
tests/
```

## Checklist config

Each item in `config/checklist.yaml` has this shape:

```yaml
- id: C05
  name: Client understanding and disclosures
  stage: 2                     # 1 facts, 2 checks, 3 decisions
  answer_type: criteria        # yes_no | category | criteria | entities | list
  applies_when:
    call_intent: [advisory, execution_only]
  depends_on: [C02, NER]
  needs_data: [product_record]
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

Starting items (fill in `sub_criteria` and `rule_queries` for each):

| ID | Name | Stage | Answer type | Applies when | Needs data |
|---|---|---|---|---|---|
| C02 | Call intent | 1 | category | always | none |
| NER | Order and entity extraction | 1 | entities | always | none |
| C01 | Identity and authority | 2 | criteria | always | client_record |
| C03 | Instruction validity | 2 | criteria | intent is execution_only or advisory with an order | none |
| C04 | Product eligibility | 2 | criteria | a product is discussed | client_record, product_record |
| C05 | Client understanding and disclosures | 2 | criteria | intent is advisory or execution_only | product_record |
| C06 | Restrictions and red flags | 2 | list | always | restricted_lists, client_record |
| C07 | Escalation | 3 | criteria | always | Stage 2 results |
| C08 | Post-call action | 3 | list | always | Stage 1 and 2 results |
| C09 | Task ownership | 3 | list | C08 has actions | ownership.yaml |

The prompt for a check only sees `question`, `sub_criteria`, the relevant rules, the transcript segments, and any context data it needs. All other fields are for the pipeline.

## Answer statuses

Use one enum everywhere:

- `PASS`
- `FAIL`
- `PARTIAL` (some sub-criteria pass, some fail)
- `NOT_APPLICABLE` (gated out)
- `CANNOT_DETERMINE` (transcript is silent, data is missing, or the LLM output failed)

`CANNOT_DETERMINE` is a valid, honest result. Never force a guess.

## Schemas (in `schemas.py`)

```python
class SubCriterionResult(BaseModel):
    id: str
    status: Status
    evidence_ids: list[str]
    note: str

class ItemResult(BaseModel):
    item_id: str
    status: Status
    confidence: float            # 0 to 1
    sub_criteria: list[SubCriterionResult]
    evidence_ids: list[str]
    rule_refs: list[str]         # IDs of rule chunks relied on
    reasoning: str               # short, for reviewers
    verified: bool = False

class RedFlag(BaseModel):
    category: Literal["aml", "sanctions", "fraud", "market_abuse",
                      "restricted_product", "suspicious_instruction", "other"]
    severity: Literal["low", "medium", "high"]
    description: str
    evidence_ids: list[str]

class Order(BaseModel):
    product_name: str | None
    product_id: str | None        # filled by normalise.py
    direction: Literal["buy", "sell"] | None
    quantity: float | None
    amount: float | None
    currency: str | None          # ISO 4217
    order_type: str | None        # market, limit, stop, and so on
    timing: str | None
    confirmed: bool               # explicitly confirmed by the client
    superseded: bool              # replaced by a later correction
    evidence_ids: list[str]

class Action(BaseModel):
    action_type: str              # key in ownership.yaml
    description: str
    owner_role: str | None        # filled by code
    evidence_ids: list[str]
```

## State (in `state.py`)

```python
class ComplianceState(TypedDict):
    call_id: str
    raw_transcript: dict
    segments: list[dict]                  # {id, speaker, role, start, end, text}
    quality_flags: list[str]
    call_intent: dict | None              # {category, confidence, evidence_ids}
    orders: list[Order]
    client_context: dict | None
    product_context: list[dict]
    applicable_items: list[str]
    results: Annotated[list[ItemResult], operator.add]
    red_flags: Annotated[list[RedFlag], operator.add]
    actions: list[Action]
    escalation: dict | None               # {required, reasons, hold_transaction}
    errors: Annotated[list[str], operator.add]
```

## Node details

**ingest:** Join Transcribe `items` with `speaker_labels.segments` by timestamp. Group into segments on speaker change or long pause. Give each segment a stable ID (`seg_0001`, and so on).

**quality_check:** Add flags for: call under a set length, average word confidence below a threshold, only one speaker. Flags do not stop the graph but are shown on the dashboard and lower confidence.

**role_inference:** One cheap LLM call on the first few segments. If the recording is dual-channel, map channels to roles in code and skip the LLM.

**call_intent:** Classify as `enquiry`, `execution_only`, or `advisory`. A call can move from enquiry to order, so classify the most regulated intent reached in the call.

**extraction:** The LLM returns a list of `Order` objects. It must capture corrections (mark older values `superseded`) and whether each order was explicitly confirmed. `normalise.py` then parses amounts ("5k" to 5000), maps currencies to ISO codes, and fuzzy-matches product names to the product table to fill `product_id`.

**context_lookup:** Plain SQL against mock tables:
- `clients`: classification, risk profile, jurisdiction, account types, pre-authorized callers, KYC status
- `products`: risk rating, allowed client types, allowed jurisdictions, complexity flag
- `restricted_lists`: sanctioned names, restricted products

If a record is missing, leave it empty. Checks that need it return `CANNOT_DETERMINE`.

**gate:** Read `applies_when` for each Stage 2 item and set `applicable_items`. Write `NOT_APPLICABLE` results for the rest.

**check_item:** One branch per applicable item via `Send`. Loads the cached rules for that item from `retrieval.py`, builds the prompt from config, and returns an `ItemResult`. C06 also returns `RedFlag` objects.

**verify:** For every `FAIL`, every red flag, and every result with confidence under 0.7, ask a second LLM call whether the cited `evidence_ids` actually support the result. If not, downgrade to `CANNOT_DETERMINE` and note why. Also check in code that every `evidence_id` exists.

**decide:**
1. Escalation: apply `escalation_rules.yaml` first. Example rules: any high-severity red flag means escalate and hold; C04 `FAIL` means hold; C01 `FAIL` means escalate. Use the LLM only when no rule fires but results are `PARTIAL` or `CANNOT_DETERMINE`.
2. Post-call actions: the LLM lists actions from confirmed orders and the transcript (place order, cancel order, send documents, obtain information, raise case).
3. Ownership: map each `action_type` to a role with `ownership.yaml` in code. The LLM never picks the owner.

**aggregate:** Make sure every checklist item has exactly one result. Missing branches become `CANNOT_DETERMINE`.

**output:** Write the call, segments, results, red flags, orders, actions, and escalation to PostgreSQL. Store rule chunks once and reference them by ID.

## Retrieval

Rules per item are fixed, so build the lookup once: run each item's `rule_queries` against ChromaDB, save the chunk IDs to a cache file, and load them at runtime. Rebuild the cache only when the rulebook or checklist changes. C06 uses one query per red flag category.

## Tests

- Unit tests for `ingest`, `normalise`, `gate`, `decide` rules, and `aggregate` (no LLM needed).
- Fixture calls in `data/fixtures/` covering: an enquiry only, an execution-only order, an advisory recommendation, a call with a correction, a call with two orders, an unconfirmed order, an ineligible product, a red flag, and an unknown caller.
- Each fixture has an `expected.json` with the correct status per item, so results can be scored.
- LLM tests use a fake `get_llm_response()` that returns saved responses.

## Build order

1. `schemas.py`, `state.py`, `config/checklist.yaml`
2. `ingest`, `quality_check`, `role_inference`, and their tests
3. `call_intent`, `extraction`, `normalise`
4. Mock tables and `context_lookup`
5. `gate`, retrieval cache, `check_item` with `Send`
6. `verify`
7. `decide` with the rule files
8. `aggregate`, `output`, `graph.py`, `main.py`
9. Fixtures and scoring script

## Open questions (check with UBS before relying on these)

- Are recordings dual-channel? If yes, drop role inference.
- Real field names for client classification, risk profile, and product eligibility.
- Which escalation rules UBS actually uses.
- Whether labelled calls will be provided for scoring.
