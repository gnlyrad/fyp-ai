"""Per-type analysis prompts, all composed from one shared base.

`ANALYSIS_SYSTEM_PROMPT` holds the rules that must never differ between types —
verbatim quoting, the evidence labels, the four verdicts. Each type appends only
the instructions that genuinely differ, so there is exactly one place to change
the auditability contract.

Two of the four types are also given regulatory extracts and so build on
`GUIDED_ANALYSIS_SYSTEM_PROMPT`, which differs from the base in one rule only.
See `nodes/guidance.py` for why deterministic items are deliberately left out.
"""
from __future__ import annotations

from .checklist import ANALYSIS_SYSTEM_PROMPT, GUIDED_ANALYSIS_SYSTEM_PROMPT

# Appended to every prompt that may receive a REGULATORY GUIDANCE section.
# The "never quote from it" rule is belt-and-braces: `locate()` searches only the
# retrieved transcript chunks, so a passage lifted from a PDF cannot become a
# span whatever the model does. The rule stops it spending a verdict on one.
GUIDANCE_RULES = """
You may also be given a REGULATORY GUIDANCE section: extracts from MAS notices
and guidelines.

- Use it to decide what counts as adequate for this question. It is the bar you
  judge against, not your own idea of good practice.
- It is NOT evidence. Never quote from it. Every `quote` you return must be
  copied from the transcript excerpts; a quote taken from the guidance is
  discarded, taking your verdict with it.
- If no guidance is supplied, or it is silent on this question, judge as you
  would without it. Do not return Unclear merely because guidance is missing.
"""

TARGETED_SYSTEM_PROMPT = GUIDED_ANALYSIS_SYSTEM_PROMPT + GUIDANCE_RULES

DETERMINISTIC_SYSTEM_PROMPT = ANALYSIS_SYSTEM_PROMPT + """
This question is a BINARY PRESENCE CHECK.

- The requirement is met if and only if an explicit statement to that effect
  appears in the excerpts. Judge presence, not quality or wording.
- If no such statement appears in the excerpts, return Fail, not Unclear. The
  excerpts are the places in the call where such a statement would be; their
  silence is the answer.
- Reserve Unclear for the case where a statement is present but you genuinely
  cannot tell whether it is the one being asked about.
- Quote the single shortest passage that settles it. One quote is enough.
"""

EXHAUSTIVE_SYSTEM_PROMPT = GUIDED_ANALYSIS_SYSTEM_PROMPT + """
This question is EXHAUSTIVE, and you are given the ENTIRE transcript in order,
not a selection of excerpts.

- Work through the transcript from start to finish and return EVERY passage that
  bears on the question, each as its own `evidence` entry. A missed occurrence is
  a wrong answer, even if you found the others.
- Do not stop at the first match, and do not summarise repeated occurrences into
  one quote. Ten occurrences means ten entries.
- If ANY occurrence is a violation, the verdict is Fail, however many compliant
  passages sit alongside it.
- Return Pass only when you have read the whole transcript and found no violating
  occurrence anywhere in it.
""" + GUIDANCE_RULES

# Not used while the pipeline is transcript-only; kept here so the prompt lands
# beside its siblings when AWS Transcribe supplies diarisation and timings.
AUDIO_METRICS_SYSTEM_PROMPT = ANALYSIS_SYSTEM_PROMPT + """
This question is about AUDIO METRICS: interruption and talk-over, silence and
hold duration, speaking pace, and tone. Judge it from the supplied speaker turns
and timings, and quote the turn that evidences the measurement.
"""

SYSTEM_PROMPT_BY_TYPE: dict[str, str] = {
    "deterministic": DETERMINISTIC_SYSTEM_PROMPT,
    "exhaustive": EXHAUSTIVE_SYSTEM_PROMPT,
    "targeted": TARGETED_SYSTEM_PROMPT,
    "audio_metrics": AUDIO_METRICS_SYSTEM_PROMPT,
}
