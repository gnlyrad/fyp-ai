"""The compliance checklist, expressed as data rather than code.

Each item is a retrieval query plus a yes/no compliance question. Keeping these
as plain dicts means the graph never changes when the checklist does.

The first eight ids match `backend/data/synthetic_conversations.json`, so the
pipeline can be scored against the existing labelled dataset. The remaining
twelve are standard advised-sales controls that the synthetic data does not yet
label; they will run and produce findings, but have no ground truth to evaluate
against.
"""
from __future__ import annotations

from ..state import ChecklistItem

CHECKLIST: list[ChecklistItem] = [
    # --- labelled in the synthetic dataset ---
    {"id": "suitability", "question": "Was suitability established before a recommendation?",
     "evidence_query": "risk preference investment horizon objectives suitability assessment"},
    {"id": "risk_ack", "question": "Did the customer acknowledge the investment risk?",
     "evidence_query": "customer understand investment can lose value accept risk"},
    {"id": "fees", "question": "Were fees and charges explained?",
     "evidence_query": "annual management fee entry charge cancellation fee cost"},
    {"id": "identity", "question": "Was identity verification completed?",
     "evidence_query": "verify identity date of birth security questions confirm details"},
    {"id": "consent", "question": "Was explicit consent recorded before acting?",
     "evidence_query": "consent authorise proceed confirm instruction agree"},
    {"id": "request", "question": "Was the service request completed and confirmed?",
     "evidence_query": "request completed confirmed next steps summary"},
    {"id": "affordability", "question": "Was affordability information adequately captured?",
     "evidence_query": "income outgoings affordability disposable budget"},
    {"id": "terms", "question": "Were key product terms explained?",
     "evidence_query": "key terms conditions product features duration"},

    # --- additional controls, unlabelled in the current dataset ---
    {"id": "vulnerability", "question": "Were signs of customer vulnerability identified and handled?",
     "evidence_query": "vulnerable health bereavement difficulty understanding support needs"},
    {"id": "cooling_off", "question": "Was the cooling-off / cancellation right explained?",
     "evidence_query": "cooling off period cancel within days right to withdraw"},
    {"id": "complaints", "question": "Was the complaints process explained when relevant?",
     "evidence_query": "complaint ombudsman escalate dissatisfied process"},
    {"id": "recording_notice", "question": "Was the customer told the call was recorded?",
     "evidence_query": "call is recorded monitoring training purposes notice"},
    {"id": "data_protection", "question": "Was personal data handled with an appropriate privacy notice?",
     "evidence_query": "data protection privacy personal information how we use your data"},
    {"id": "pressure", "question": "Did the advisor avoid pressure selling or undue urgency?",
     "evidence_query": "offer expires today must decide now limited time hurry"},
    {"id": "guarantees", "question": "Did the advisor avoid guaranteeing returns or outcomes?",
     "evidence_query": "guaranteed return promise cannot lose sure thing"},
    {"id": "alternatives", "question": "Were alternative options or products discussed?",
     "evidence_query": "alternative option other products compare instead"},
    {"id": "questions", "question": "Was the customer given an opportunity to ask questions?",
     "evidence_query": "any questions anything else unclear happy to explain"},
    {"id": "next_steps", "question": "Were next steps and timescales clearly confirmed?",
     "evidence_query": "next steps timescale working days you will receive"},
    {"id": "documentation", "question": "Was the customer told what documentation they would receive?",
     "evidence_query": "send you documents confirmation in writing paperwork post email"},
    {"id": "advice_scope", "question": "Was the scope of advice (advised vs execution-only) made clear?",
     "evidence_query": "advised execution only recommendation personal advice scope"},
]

CHECKLIST_BY_ID = {item["id"]: item for item in CHECKLIST}

ANALYSIS_SYSTEM_PROMPT = """\
You are a compliance analyst reviewing a recorded customer call.

You will be given ONE compliance question and EXCERPTS from the call transcript.

Rules:
- Answer using the transcript excerpts ONLY. Never rely on outside knowledge.
- `evidence_quote` MUST be copied verbatim from the excerpts. Do not paraphrase,
  reformat, or repair it. If no excerpt supports an answer, return an empty quote.
- Choose exactly one verdict:
    Pass    - the transcript shows the requirement was met
    Fail    - the transcript shows the requirement was not met
    Unclear - the excerpts do not settle the question
    N/A     - the requirement does not apply to this kind of call
- Prefer Unclear over guessing. An unsupported Pass is a serious error.
"""
