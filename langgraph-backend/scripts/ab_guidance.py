"""A/B the regulatory grounding: judge each item twice, with and without guidance.

Everything else is held constant — same transcript, same chunks, same retrieved
evidence, same prompt, same model at temperature 0. The only variable is whether
the REGULATORY GUIDANCE block is present in the human message, which is exactly
the change `nodes/guidance.py` introduced.

Only `targeted` and `exhaustive` items are run: they are the only ones the
pipeline grounds, so they are the only ones where an A/B means anything.

    python scripts/ab_guidance.py requests/ab_request.json [--json out.json]

Results go to stdout. This deliberately stops before `output_node`, so there is
no `report` and nothing is written to `output/` unless `--json` is given.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compliance_pipeline.graph import ANALYSIS_NODES
from compliance_pipeline.nodes.classification import classification_node
from compliance_pipeline.nodes.applicability import applicability_node
from compliance_pipeline.nodes.guidance import GUIDED_TYPES, guidance_node
from compliance_pipeline.nodes.ingestion import ingestion_node
from compliance_pipeline.nodes.retrieval import retrieval_node
from compliance_pipeline.state import DEFAULT_QUESTION_TYPE

NODE_FN = {qtype: fn for qtype, (_, fn) in ANALYSIS_NODES.items()}

# Piping this into anything block-buffers stdout, which hides a run that takes
# minutes. Flush every line so progress is visible wherever it is sent.
say = functools.partial(print, flush=True)


def judge(item, state, guidance):
    """One analysis node call, routed the way the graph would route it."""
    qtype = item.get("question_type") or DEFAULT_QUESTION_TYPE
    task = {
        "item": item,
        "retrieved": state["evidence"].get(item["id"], []),
        "transcript": state["transcript"],
        "guidance": guidance,
        "chunks": state["chunks"],
        "applicability": state["applicability"].get(item["id"], {}),
        "retrieval_diagnostics": state["retrieval_diagnostics"].get(item["id"], {}),
    }
    return NODE_FN[qtype](task)["findings"][0]


def cites(passages):
    return ", ".join(f"{p['source']} p.{p['page']}" for p in passages)


def main(request_path: str, json_out: str | None) -> None:
    payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
    state = payload.get("input", payload)

    # Run the pipeline up to (but not including) the analysis fan-out, so both
    # arms are judged against byte-identical evidence. This prelude is slow —
    # one classification call plus an embedding call per guided item — so it
    # reports as it goes rather than sitting silent.
    say("ingestion...")
    state |= ingestion_node(state)
    say("classification...")
    state |= classification_node(state)
    say("applicability...")
    state |= applicability_node(state)
    say("guidance (one corpus lookup per guided item)...")
    state |= guidance_node(state)
    say("retrieval...")
    state |= retrieval_node(state)

    items = [
        item for item in state["checklist"]
        if (item.get("question_type") or DEFAULT_QUESTION_TYPE) in GUIDED_TYPES
    ]
    total = len(state["checklist"])
    say(f"\n{len(items)} guided items ({total} in checklist, "
        f"{total - len(items)} ungrounded by design)\n")

    rows = []
    for index, item in enumerate(items, start=1):
        passages = state["guidance"].get(item["id"], [])
        without = judge(item, state, [])
        with_ = judge(item, state, passages)
        moved = without["verdict"] != with_["verdict"]
        rows.append((item, without, with_, passages, moved))

        flag = "CHANGED" if moved else "same"
        say(f"[{flag:7}] {index}/{len(items)} {item['id']}  ({item.get('question_type')})")
        say(f"    without : {without['verdict']:8} {without['reason']}")
        say(f"      quote : {str(without.get('evidence_quote'))[:88]}")
        say(f"    with    : {with_['verdict']:8} {with_['reason']}")
        say(f"      quote : {str(with_.get('evidence_quote'))[:88]}")
        if passages:
            say(f"      basis : {cites(passages)}")
        say()

    changed = sum(1 for *_, moved in rows if moved)
    quote_moves = sum(
        1 for _, w, g, _, _ in rows if w.get("evidence_quote") != g.get("evidence_quote")
    )
    # An arm that errored was forced to Unclear by the error path, not judged.
    # Counting those as evidence of anything would overstate the result.
    errored = sum(
        1 for _, w, g, _, _ in rows
        if "analysis_unavailable" in {w.get("evidence_status"), g.get("evidence_status")}
    )
    say(f"verdicts changed: {changed}/{len(rows)}   unchanged: {len(rows) - changed}")
    say(f"evidence quote changed: {quote_moves}/{len(rows)}")
    if errored:
        say(f"WARNING: {errored} item(s) had a model error in one arm — not a result")

    if json_out:
        Path(json_out).write_text(json.dumps([
            {
                "checklist_id": item["id"],
                "question_type": item.get("question_type"),
                "verdict_without": w["verdict"],
                "verdict_with": g["verdict"],
                "changed": moved,
                "reason_without": w["reason"],
                "reason_with": g["reason"],
                "quote_without": w.get("evidence_quote"),
                "quote_with": g.get("evidence_quote"),
                "regulatory_basis": p,
            }
            for item, w, g, p, moved in rows
        ], indent=2), encoding="utf-8")
        say(f"\nwrote {json_out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    out = None
    if "--json" in args:
        i = args.index("--json")
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    main(args[0] if args else "requests/ab_request.json", out)
