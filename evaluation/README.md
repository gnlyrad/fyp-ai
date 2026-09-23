# Offline NLP evaluation scaffold

This directory contains the committed evaluation data and the small loader used
by the future offline evaluator. It deliberately does not regenerate synthetic
calls, make HTTP requests, call a model, score predictions, or write results.

## Dataset

`datasets/pilot/` is a generated snapshot from `fyp-be`. Do not hand-edit it.
Correct the source in `fyp-be`, rerun the exporter, and commit the resulting
snapshot update here. `manifest.json` identifies its source commit and hashes.

## Direct graph input

The LangGraph graph's `AnalyseInput` has three relevant fields:

```python
{
    "results": transcript_json,
    "speaker_roles": speaker_roles_json,
    "checklist": [{"id": item["id"], "question": item["text"]} for item in checklist_json],
}
```

`transcript_json` is the complete content of a call's `transcript.json`; it is
the **AWS Transcribe `results` object**, not an outer Transcribe job envelope.
It contains `transcripts` and `audio_segments`. The graph accepts this under
the `results` key. The graph uses `audio_segments` in preference to
`transcripts`.

The saved `checklist.json` is authoritative evaluation data, but it is not
passed unchanged: its `text` fields are deterministically projected to the
graph's `question` fields. `ground_truth.json` and `manifest.json` never enter
the graph; they are for later scoring and provenance.

## LangSmith Studio

Studio exposes the graph input fields separately. For one call:

| Studio field | Paste |
| --- | --- |
| Transcript | Leave empty |
| Results | The complete `transcript.json` object |
| Audio Segments | Leave empty; they are already inside Results |
| Checklist | The projected list of `{ "id", "question" }` objects |
| Speaker Roles | The complete `speaker_roles.json` mapping, for example `{ "spk_0": "RM", "spk_1": "Customer" }` |

Studio is useful for a manual smoke run and tracing a graph. The future batch
evaluator should import and call `graph.invoke(...)` directly, avoiding HTTP
serialization and testing the actual graph implementation.

## Validate a prepared input

Run from the repository root:

```bat
python -B evaluation/scripts/prepare_run.py --call CALL_001 --check-only
```

It validates the dataset inventory and prints the selected call's direct graph
input without calling LangGraph or an LLM. `--all --check-only` validates every
call. A later runner will import `evaluation.fyp_evaluation_loader` and call the
real graph; scoring and result persistence intentionally remain future work.
