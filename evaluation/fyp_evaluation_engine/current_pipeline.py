"""Bridge the staged pipeline to the published pilot, without sending gold labels.

Supplied checklist IDs are kept exactly as provided. Evidence is mapped only
after the full sequence of ingested turns
matches the published transcript (text and speaker); IDs are never guessed.
"""
from .persistence import to_jsonable

STATUS = {"PASS": "Pass", "FAIL": "Fail", "PARTIAL": "Fail",
          "NOT_APPLICABLE": "N/A", "CANNOT_DETERMINE": "Unclear"}


def prepare_input(value):
    return {**value, "checklist": [
        {"id": item["id"], "question": item["question"],
         **({"requirement_type": item["requirement_type"]}
            if item.get("requirement_type") in {"OBLIGATION", "PROHIBITION"} else {})}
        for item in value["checklist"]]}


def adapt_output(output, transcript):
    output = to_jsonable(output)
    groups = []
    for source in transcript["audio_segments"]:
        speaker = source.get("speaker_label", "unknown")
        text = source.get("transcript", "").strip()
        if not groups or groups[-1]["speaker"] != speaker:
            groups.append({"speaker": speaker, "text": "", "spans": []})
        group = groups[-1]
        start = len(group["text"]) + bool(group["text"] and text)
        group["text"] = (group["text"] + " " + text).strip()
        group["spans"].append((source["id"], start, start + len(text)))
    segments = output.get("segments", [])
    aligned = len(segments) == len(groups) and all(
        seg.get("text") == group["text"] and seg.get("speaker") == group["speaker"]
        for seg, group in zip(segments, groups))
    ids = [seg.get("id") for seg in segments]
    aligned = aligned and len(ids) == len(set(ids))
    mapping = dict(zip(ids, groups)) if aligned else {}
    findings, chunks = [], []
    for result in output.get("final_results", []):
        if not isinstance(result, dict):
            findings.append(result)
            continue
        item_id = result.get("item_id", "")
        spans = []
        evidence = result.get("evidence")
        if not isinstance(evidence, list):
            spans = None
        else:
            for ref in evidence:
                group = mapping.get(ref.get("segment_id")) if isinstance(ref, dict) else None
                start = ref.get("start_char") if isinstance(ref, dict) else None
                end = ref.get("end_char") if isinstance(ref, dict) else None
                if (not group or type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(group["text"])):
                    spans.append({"chunk_id": "invalid-native-reference"})
                    continue
                overlaps = [(sid, max(start, a), min(end, b)) for sid, a, b in group["spans"]
                            if max(start, a) < min(end, b)]
                if not overlaps:
                    spans.append({"chunk_id": "invalid-native-reference"})
                for sid, a, b in overlaps:
                    chunk_id = f"source-{len(chunks)}"
                    chunks.append({"chunk_id": chunk_id, "source_segment_id": sid})
                    spans.append({"chunk_id": chunk_id, "text": group["text"][a:b]})
        findings.append({
            "checklist_id": item_id,
            "verdict": STATUS.get(result.get("status"), result.get("status")),
            "native_status": result.get("status"), "spans": spans,
            "reason": result.get("reasoning"),
            "execution_status": result.get("execution_status", "success"),
            "failure_category": result.get("failure_category"),
            "failure_stage": result.get("failure_stage"),
        })
    return {"report": {"findings": findings, "summary": output.get("summary"),
                        "warnings": output.get("errors", [])}, "chunks": chunks}
