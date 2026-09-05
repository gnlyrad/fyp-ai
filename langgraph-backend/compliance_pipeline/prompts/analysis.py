"""Per-type analysis prompts, all composed from one shared base.

`ANALYSIS_SYSTEM_PROMPT` holds the rules that must never differ between types —
verbatim quoting, the evidence labels, the four verdicts. Each type appends only
the instructions that genuinely differ, so there is exactly one place to change
the auditability contract.
"""
from __future__ import annotations

from .checklist import ANALYSIS_SYSTEM_PROMPT

TARGETED_SYSTEM_PROMPT = ANALYSIS_SYSTEM_PROMPT

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

EXHAUSTIVE_SYSTEM_PROMPT = ANALYSIS_SYSTEM_PROMPT + """
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
"""

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
