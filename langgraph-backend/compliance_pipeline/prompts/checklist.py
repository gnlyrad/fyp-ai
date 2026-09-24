"""The compliance checklist, expressed as data rather than code.

Each item is a retrieval query plus a yes/no compliance question, tagged with the
`question_type` that decides how much transcript it is given and which analysis
node judges it. Keeping these as plain dicts means the graph never changes when
the checklist does.

Tagging them here means the built-in checklist skips the classification node
entirely; classification exists for caller-supplied items, which arrive as bare
{id, question} pairs.

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
     "evidence_query": "risk preference investment horizon objectives suitability assessment",
     "question_type": "targeted"},
    {"id": "risk_ack", "question": "Did the customer acknowledge the investment risk?",
     "evidence_query": "customer understand investment can lose value accept risk",
     "question_type": "targeted"},
    {"id": "fees", "question": "Were fees and charges explained?",
     "evidence_query": "annual management fee entry charge cancellation fee cost",
     "question_type": "targeted"},
    {"id": "identity", "question": "Was identity verification completed?",
     "evidence_query": "verify identity date of birth security questions confirm details",
     "question_type": "deterministic"},
    {"id": "consent", "question": "Was explicit consent recorded before acting?",
     "evidence_query": "consent authorise proceed confirm instruction agree",
     "question_type": "deterministic"},
    {"id": "request", "question": "Was the service request completed and confirmed?",
     "evidence_query": "request completed confirmed next steps summary",
     "question_type": "targeted"},
    {"id": "affordability", "question": "Was affordability information adequately captured?",
     "evidence_query": "income outgoings affordability disposable budget",
     "question_type": "targeted"},
    {"id": "terms", "question": "Were key product terms explained?",
     "evidence_query": "key terms conditions product features duration",
     "question_type": "targeted"},

    # --- additional controls, unlabelled in the current dataset ---
    {"id": "vulnerability", "question": "Were signs of customer vulnerability identified and handled?",
     "evidence_query": "vulnerable health bereavement difficulty understanding support needs",
     "question_type": "targeted"},
    {"id": "cooling_off", "question": "Was the cooling-off / cancellation right explained?",
     "evidence_query": "cooling off period cancel within days right to withdraw",
     "question_type": "targeted"},
    {"id": "complaints", "question": "Was the complaints process explained when relevant?",
     "evidence_query": "complaint ombudsman escalate dissatisfied process",
     "question_type": "targeted"},
    {"id": "recording_notice", "question": "Was the customer told the call was recorded?",
     "evidence_query": "call is recorded monitoring training purposes notice",
     "question_type": "deterministic"},
    {"id": "data_protection", "question": "Was personal data handled with an appropriate privacy notice?",
     "evidence_query": "data protection privacy personal information how we use your data",
     "question_type": "deterministic"},
    {"id": "pressure", "question": "Did the advisor avoid pressure selling or undue urgency?",
     "evidence_query": "offer expires today must decide now limited time hurry",
     "question_type": "exhaustive"},
    {"id": "guarantees", "question": "Did the advisor avoid guaranteeing returns or outcomes?",
     "evidence_query": "guaranteed return promise cannot lose sure thing",
     "question_type": "exhaustive"},
    {"id": "alternatives", "question": "Were alternative options or products discussed?",
     "evidence_query": "alternative option other products compare instead",
     "question_type": "targeted"},
    {"id": "questions", "question": "Was the customer given an opportunity to ask questions?",
     "evidence_query": "any questions anything else unclear happy to explain",
     "question_type": "deterministic"},
    {"id": "next_steps", "question": "Were next steps and timescales clearly confirmed?",
     "evidence_query": "next steps timescale working days you will receive",
     "question_type": "targeted"},
    {"id": "documentation", "question": "Was the customer told what documentation they would receive?",
     "evidence_query": "send you documents confirmation in writing paperwork post email",
     "question_type": "targeted"},
    {"id": "advice_scope", "question": "Was the scope of advice (advised vs execution-only) made clear?",
     "evidence_query": "advised execution only recommendation personal advice scope",
     "question_type": "targeted"},
]

CHECKLIST_BY_ID = {item["id"]: item for item in CHECKLIST}

# The one rule that differs between the guided and unguided bases. Regulation is
# outside knowledge, so an item that is given regulatory extracts cannot be told
# to ignore outside knowledge — but the substitute has to be just as strict about
# where *facts* may come from, because `locate()` and the whole audit trail rest
# on every quote being transcript text.
_EVIDENCE_ONLY_RULE = (
    "- Answer using the transcript excerpts ONLY. Never rely on outside knowledge."
)
_GUIDED_EVIDENCE_RULE = (
    "- The REGULATORY GUIDANCE decides the standard you apply; the transcript\n"
    "  excerpts are the ONLY source of fact about what happened on this call.\n"
    "  Never treat the guidance as something the advisor said or did."
)

_ANALYSIS_BASE = """\
You are a compliance analyst reviewing a recorded customer call.

You will be given ONE compliance question and EXCERPTS from the call transcript.

Rules:
{evidence_rule}
- Return EVERY passage in the excerpts that bears on the question, as a separate
  entry in `evidence`. Label each one:
    compliant - the passage shows the requirement being met
    violation - the passage shows it being breached, or shows prohibited conduct
- Each `quote` MUST be copied verbatim from the excerpts, character for character.
  Do not paraphrase, reformat, trim, or repair it, and do not merge two separate
  passages into one quote. A quote that does not appear exactly in the excerpts is
  discarded, taking your verdict with it.
- Quote the shortest passage that makes the point; do not quote a whole excerpt.
- If no excerpt supports an answer, return an empty `evidence` list.
- Choose exactly one verdict:
    Pass    - the transcript shows the requirement was met
    Fail    - the transcript shows the requirement was not met
    Unclear - the excerpts do not settle the question
    N/A     - the requirement does not apply to this kind of call
- Prefer Unclear over guessing. An unsupported Pass is a serious error.
"""

# Byte-identical to what deterministic and audio-metrics items have always been
# sent; the split above must not move them.
ANALYSIS_SYSTEM_PROMPT = _ANALYSIS_BASE.format(evidence_rule=_EVIDENCE_ONLY_RULE)

# For items that are handed regulatory extracts alongside the transcript.
GUIDED_ANALYSIS_SYSTEM_PROMPT = _ANALYSIS_BASE.format(evidence_rule=_GUIDED_EVIDENCE_RULE)
