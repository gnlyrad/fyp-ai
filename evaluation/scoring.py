"""Score compliance-pipeline outputs against fixture ground truth, and
aggregate the scores. Pure functions, shared by `upload_outputs.py` (scores
each output before upload) and `dashboard.py` (aggregates stored scores).

Ground truth is `langgraph-backend/compliance_pipeline/data/fixtures/
<call_id>/expected.json`. Each checklist item lists one or more acceptable
statuses; an item matches when the actual status is one of them (the same
rule as `langgraph-backend/scripts/score_fixtures.py`).

Breach metrics follow the evaluation service's conventions: FAIL (a breach)
is the positive class, PARTIAL counts as a breach, CANNOT_DETERMINE is an
abstention, and a ratio with a zero denominator has value None, never 0.
An item whose acceptable statuses mix breach and non-breach values (e.g.
["PASS", "CANNOT_DETERMINE"]) has no binary gold label and is left out of
the binary metrics; it still counts for status accuracy.
"""
from __future__ import annotations

from collections import Counter, defaultdict

SCORER_VERSION = "fixture-scorer-v1"
STATUSES = ["PASS", "PARTIAL", "FAIL", "NOT_APPLICABLE", "CANNOT_DETERMINE"]
BREACH_STATUSES = {"FAIL", "PARTIAL"}

OUTCOMES = ["decision", "abstention", "not_applicable", "missing"]
BREACH_OUTCOMES = ["detected", "false_clearance", "abstained", "not_applicable", "missing"]


def ratio(numerator: float, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


# ---------------------------------------------------------------------------
# Scoring one output.
# ---------------------------------------------------------------------------

def _acceptable(value) -> list[str]:
    return [value] if isinstance(value, str) else list(value or [])


def gold_binary(acceptable: list[str]) -> str | None:
    """FAIL when every acceptable status is a breach, PASS when only PASS is
    acceptable, otherwise None (no binary gold label)."""
    if acceptable and set(acceptable) <= BREACH_STATUSES:
        return "FAIL"
    if set(acceptable) == {"PASS"}:
        return "PASS"
    return None


def outcome(actual: str | None) -> str:
    if actual is None:
        return "missing"
    if actual == "CANNOT_DETERMINE":
        return "abstention"
    if actual == "NOT_APPLICABLE":
        return "not_applicable"
    return "decision"


def predicted_binary(actual: str | None) -> str | None:
    if actual == "PASS":
        return "PASS"
    if actual in BREACH_STATUSES:
        return "FAIL"
    return None


def score_item(item_id: str, expected_value, actual: str | None) -> dict:
    acceptable = _acceptable(expected_value)
    gold = gold_binary(acceptable)
    pred = predicted_binary(actual)
    item_outcome = outcome(actual)

    confusion = None
    if gold and pred:
        confusion = {("FAIL", "FAIL"): "TP", ("PASS", "PASS"): "TN",
                     ("PASS", "FAIL"): "FP", ("FAIL", "PASS"): "FN"}[(gold, pred)]

    breach_outcome = None
    if gold == "FAIL":
        breach_outcome = {"abstention": "abstained", "not_applicable": "not_applicable",
                          "missing": "missing"}.get(item_outcome)
        if breach_outcome is None:
            breach_outcome = "detected" if pred == "FAIL" else "false_clearance"

    return {
        "item_id": item_id,
        "expected": acceptable,
        "actual": actual,
        "match": actual in acceptable,
        "gold_binary": gold,
        "outcome": item_outcome,
        "confusion": confusion,
        "breach_outcome": breach_outcome,
    }


def _set_score(expected: set, actual: set) -> dict:
    hits = len(expected & actual)
    return {
        "expected": sorted(expected),
        "actual": sorted(actual),
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
        "hits": hits,
        "match": expected <= actual,
    }


def score_output(expected: dict, payload: dict) -> dict:
    """Score one pipeline output (`build_payload` shape) against its fixture."""
    actual_by_id = {r.get("item_id"): r.get("status") for r in payload.get("results") or []}
    items = [score_item(item_id, value, actual_by_id.get(item_id))
             for item_id, value in (expected.get("items") or {}).items()]

    actual_intent = (payload.get("call_intent") or {}).get("category")
    exp_esc = bool(expected.get("escalation_required", False))
    act_esc = bool((payload.get("escalation") or {}).get("required", False))

    return {
        "scorer_version": SCORER_VERSION,
        "items": items,
        "item_accuracy": ratio(sum(i["match"] for i in items), len(items)),
        "call_intent": {"expected": expected.get("call_intent"), "actual": actual_intent,
                        "match": actual_intent == expected.get("call_intent")},
        "escalation": {"expected": exp_esc, "actual": act_esc, "match": exp_esc == act_esc,
                       "confusion": {(True, True): "TP", (False, False): "TN",
                                     (False, True): "FP", (True, False): "FN"}[(exp_esc, act_esc)]},
        "action_types": _set_score(set(expected.get("expected_action_types") or []),
                                   {a.get("action_type") for a in payload.get("actions") or []} - {None}),
        "red_flag_categories": _set_score(set(expected.get("expected_red_flag_categories") or []),
                                          {f.get("category") for f in payload.get("red_flags") or []} - {None}),
    }


# ---------------------------------------------------------------------------
# Aggregation over scored runs.
# ---------------------------------------------------------------------------

def aggregate_items(item_scores: list[dict]) -> dict:
    """Checklist-item metrics over any set of item scores (overall, one item, one call)."""
    n = len(item_scores)
    outcomes = Counter(i["outcome"] for i in item_scores)
    confusion = Counter(i["confusion"] for i in item_scores if i["confusion"])
    breach = Counter(i["breach_outcome"] for i in item_scores if i["breach_outcome"])
    gold_fail = sum(1 for i in item_scores if i["gold_binary"] == "FAIL")
    tp, tn, fp, fn = (confusion[k] for k in ("TP", "TN", "FP", "FN"))
    decisions = tp + tn + fp + fn
    status_matrix: dict[str, Counter] = defaultdict(Counter)
    for i in item_scores:
        status_matrix[i["expected"][0] if i["expected"] else "?"][i["actual"] or "missing"] += 1

    return {
        "total": n,
        "status_accuracy": ratio(sum(i["match"] for i in item_scores), n),
        "status_confusion_matrix": {k: dict(v) for k, v in status_matrix.items()},
        "outcome_population": {o: outcomes.get(o, 0) for o in OUTCOMES},
        "decision_coverage": ratio(outcomes["decision"], n),
        "abstention_rate": ratio(outcomes["abstention"], n),
        "not_applicable_rate": ratio(outcomes["not_applicable"], n),
        "missing_rate": ratio(outcomes["missing"], n),
        "binary": {
            "gold_labelled": sum(1 for i in item_scores if i["gold_binary"]),
            "no_binary_gold": sum(1 for i in item_scores if not i["gold_binary"]),
            "decisions": decisions,
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
            "accuracy": ratio(tp + tn, decisions),
            "breach_precision": ratio(tp, tp + fp),
            "breach_recall": ratio(tp, tp + fn),
        },
        "breach_outcomes": {
            "gold_fail_total": gold_fail,
            "counts": {o: breach.get(o, 0) for o in BREACH_OUTCOMES},
            "automatic_breach_detection_rate": ratio(breach["detected"], gold_fail),
            "false_clearance_rate": ratio(breach["false_clearance"], gold_fail),
            "abstention_rate": ratio(breach["abstained"], gold_fail),
        },
    }


def _set_metrics(scores: list[dict]) -> dict:
    expected = sum(len(s["expected"]) for s in scores)
    actual = sum(len(s["actual"]) for s in scores)
    hits = sum(s["hits"] for s in scores)
    return {
        "recall": ratio(hits, expected),
        "precision": ratio(hits, actual),
        "all_expected_found": ratio(sum(s["match"] for s in scores), len(scores)),
    }


def aggregate_calls(scores: list[dict]) -> dict:
    """Call-level metrics: intent, escalation, actions, red flags."""
    n = len(scores)
    esc = Counter(s["escalation"]["confusion"] for s in scores)
    tp, tn, fp, fn = (esc[k] for k in ("TP", "TN", "FP", "FN"))
    return {
        "calls": n,
        "call_intent_accuracy": ratio(sum(s["call_intent"]["match"] for s in scores), n),
        "escalation": {
            "accuracy": ratio(tp + tn, n),
            "recall": ratio(tp, tp + fn),
            "precision": ratio(tp, tp + fp),
            "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        },
        "action_types": _set_metrics([s["action_types"] for s in scores]),
        "red_flag_categories": _set_metrics([s["red_flag_categories"] for s in scores]),
    }


def aggregate(scored_runs: list[dict]) -> dict | None:
    """`scored_runs`: docs with `call_id` and `score`. None when nothing is scored."""
    if not scored_runs:
        return None
    all_items = [i for r in scored_runs for i in r["score"]["items"]]
    by_item: dict[str, list[dict]] = defaultdict(list)
    for i in all_items:
        by_item[i["item_id"]].append(i)
    by_call: dict[str, list[dict]] = defaultdict(list)
    for r in scored_runs:
        by_call[r["call_id"]].append(r)
    return {
        "items": aggregate_items(all_items),
        "calls": aggregate_calls([r["score"] for r in scored_runs]),
        "by_item": {k: aggregate_items(v) for k, v in sorted(by_item.items())},
        "by_call": {
            k: {"items": aggregate_items([i for r in v for i in r["score"]["items"]]),
                "calls": aggregate_calls([r["score"] for r in v])}
            for k, v in sorted(by_call.items())
        },
    }


# ---------------------------------------------------------------------------
# Metric definitions (dashboard tooltips).
# ---------------------------------------------------------------------------

METRIC_DEFINITIONS = {
    "status_accuracy": {
        "label": "Status accuracy",
        "population": "Every checklist item in scored runs.",
        "formula": "items whose actual status is one of the expected acceptable statuses / all items",
        "interpretation": "Headline agreement with the fixture ground truth. Higher is better.",
        "watch_out": "Items that accept several statuses are easier to match; read it with the breach metrics.",
    },
    "decision_coverage": {
        "label": "Decision coverage",
        "population": "Every checklist item in scored runs.",
        "formula": "items answered PASS, PARTIAL or FAIL / all items",
        "interpretation": "How often the pipeline commits to a decision.",
        "watch_out": "Says nothing about whether the decisions were correct.",
    },
    "abstention_rate": {
        "label": "Abstention rate",
        "population": "Every checklist item in scored runs.",
        "formula": "items answered CANNOT_DETERMINE / all items",
        "interpretation": "How often the pipeline declines to decide.",
    },
    "not_applicable_rate": {
        "label": "Not-applicable rate",
        "population": "Every checklist item in scored runs.",
        "formula": "items answered NOT_APPLICABLE / all items",
        "interpretation": "Items the pipeline's gate judged out of scope for the call.",
    },
    "missing_rate": {
        "label": "Missing result rate",
        "population": "Every checklist item in scored runs.",
        "formula": "expected items with no result in the output / all items",
        "interpretation": "Operational reliability. Expected to be 0.",
    },
    "binary_accuracy": {
        "label": "Binary accuracy",
        "population": "Items with a binary gold label and a PASS/FAIL/PARTIAL answer.",
        "formula": "(TP + TN) / (TP + TN + FP + FN)",
        "interpretation": "Correctness when the pipeline makes a definite decision.",
        "watch_out": "Excludes abstentions, N/A and items whose gold accepts mixed statuses. Read it with decision coverage.",
    },
    "breach_precision": {
        "label": "Breach precision",
        "population": "Items with a binary gold label and a PASS/FAIL/PARTIAL answer.",
        "formula": "TP / (TP + FP)",
        "interpretation": "Among breaches flagged (FAIL/PARTIAL), the share that are real breaches.",
        "watch_out": "n/a when the pipeline flagged no breach.",
    },
    "breach_recall": {
        "label": "Breach recall",
        "population": "Items with a binary gold label and a PASS/FAIL/PARTIAL answer.",
        "formula": "TP / (TP + FN)",
        "interpretation": "Among real breaches the pipeline decided on, the share flagged.",
        "watch_out": "Excludes abstained breaches. Use the breach detection rate for every gold breach.",
    },
    "automatic_breach_detection_rate": {
        "label": "Breach detection rate",
        "population": "Every item whose gold is a breach (FAIL/PARTIAL only).",
        "formula": "gold breaches answered FAIL or PARTIAL / all gold breaches",
        "interpretation": "End-to-end share of real breaches surfaced. Higher is better.",
        "watch_out": "Abstentions and N/A stay in the denominator, so this is stricter than breach recall.",
    },
    "false_clearance_rate": {
        "label": "False-clearance rate",
        "population": "Every item whose gold is a breach (FAIL/PARTIAL only).",
        "formula": "gold breaches answered PASS / all gold breaches",
        "interpretation": "Share of real breaches wrongly cleared. Lower is better; the costliest error.",
    },
    "breach_abstention_rate": {
        "label": "Breach abstention rate",
        "population": "Every item whose gold is a breach (FAIL/PARTIAL only).",
        "formula": "gold breaches answered CANNOT_DETERMINE / all gold breaches",
        "interpretation": "Share of real breaches the pipeline declined to decide.",
    },
    "call_intent_accuracy": {
        "label": "Call intent accuracy",
        "population": "Every scored call.",
        "formula": "calls whose call_intent.category equals the expected intent / calls",
        "interpretation": "Stage 1 classification correctness (enquiry / execution_only / advisory).",
    },
    "escalation_accuracy": {
        "label": "Escalation accuracy",
        "population": "Every scored call.",
        "formula": "calls whose escalation.required equals escalation_required / calls",
        "interpretation": "Agreement on whether the call needs escalation.",
    },
    "escalation_recall": {
        "label": "Escalation recall",
        "population": "Calls that should be escalated.",
        "formula": "TP / (TP + FN), escalation required is the positive class",
        "interpretation": "Share of calls needing escalation that were escalated. Missing one is the costly error.",
    },
    "escalation_precision": {
        "label": "Escalation precision",
        "population": "Calls the pipeline escalated.",
        "formula": "TP / (TP + FP)",
        "interpretation": "Share of escalations that were warranted. Low precision means review noise.",
    },
    "action_recall": {
        "label": "Action-type recall",
        "population": "Expected follow-up action types across scored calls.",
        "formula": "expected action types found / expected action types (micro)",
        "interpretation": "Whether the pipeline raises the follow-ups it should.",
    },
    "action_precision": {
        "label": "Action-type precision",
        "population": "Action types the pipeline raised across scored calls.",
        "formula": "expected action types found / action types raised (micro)",
        "interpretation": "Share of raised action types that were expected.",
        "watch_out": "Fixtures list the actions that must appear, not every acceptable one, so extra actions are not always wrong.",
    },
    "red_flag_recall": {
        "label": "Red-flag recall",
        "population": "Expected red-flag categories across scored calls.",
        "formula": "expected categories found / expected categories (micro)",
        "interpretation": "Whether real red flags (AML, suspicious instructions, ...) are caught.",
        "watch_out": "n/a when no scored call expects a red flag.",
    },
    "red_flag_precision": {
        "label": "Red-flag precision",
        "population": "Red-flag categories the pipeline raised across scored calls.",
        "formula": "expected categories found / categories raised (micro)",
        "interpretation": "Share of raised red-flag categories that were expected. Low precision means false alarms.",
    },
}
