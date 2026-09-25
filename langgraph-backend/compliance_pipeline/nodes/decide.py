"""`decide`: escalation (C07), post-call actions (C08), task ownership (C09).

Runs once after `verify` joins, before `aggregate`. Code first throughout
(CLAUDE.md "Prefer plain code over an LLM whenever the task is
deterministic"):

1. Escalation: evaluate `config/escalation_rules.yaml` in order. Every
   firing rule contributes -- escalate/hold are ORed across fired rules,
   and every fired rule's reason and id are kept. The LLM is used only as a
   fallback, when no rule fired at all but some Stage 2 result is PARTIAL
   or CANNOT_DETERMINE (an unclear case the rules don't cover).
2. Post-call actions: one LLM call (when there are transcript segments)
   proposing actions with short verbatim evidence quotes, resolved to
   persisted refs in code via `resolve_evidence`. If escalation ended up
   required and the LLM did not itself propose a `raise_case`, code adds
   one so a case always gets raised when escalation is required.
3. Ownership: `action_type -> owner_role` is a plain lookup in
   `config/ownership.yaml`; the LLM never picks the owner.

Never raises: any unexpected failure falls back to CANNOT_DETERMINE C07/C08
results, a NOT_APPLICABLE C09, no actions, and an `errors` entry.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, create_model

from compliance_pipeline import config
from compliance_pipeline.llm import get_llm_response
from compliance_pipeline.nodes._common import (
    format_transcript,
    merged_results,
    resolve_evidence,
    rollup_status,
)
from compliance_pipeline.prompts.decide import (
    SYSTEM_ACTIONS,
    SYSTEM_ESCALATION,
    build_actions_user,
    build_escalation_user,
)
from compliance_pipeline.schemas import (
    Action,
    Escalation,
    EscalationDecision,
    EvidenceQuote,
    EvidenceRef,
    ItemResult,
    RedFlag,
    Status,
    SubCriterionResult,
)
from compliance_pipeline.state import ComplianceState

# The only `when` keys this rule engine knows how to evaluate. Validated at
# load, like `config.py`'s `_validate_applies_when` for checklist items.
_KNOWN_WHEN_KEYS = {"red_flag_severity", "item", "status", "other_item_status"}


# ---------------------------------------------------------------------------
# Step 1: escalation rules.
# ---------------------------------------------------------------------------

def _validate_rule(rule: dict) -> None:
    when = rule.get("when") or {}
    unknown = set(when) - _KNOWN_WHEN_KEYS
    if unknown:
        raise ValueError(
            f"escalation_rules.yaml rule {rule.get('id')!r} has unknown when "
            f"key(s): {sorted(unknown)}; expected subset of {sorted(_KNOWN_WHEN_KEYS)}"
        )


def _escalation_rules() -> list[dict]:
    """Escalation rules from config, validated here (config.py only
    validates checklist `applies_when`). Not cached itself so a test can
    monkeypatch `config.load_escalation_rules` directly."""
    rules = config.load_escalation_rules()
    for rule in rules:
        _validate_rule(rule)
    return rules


def _dedupe_refs(refs: list[EvidenceRef]) -> list[EvidenceRef]:
    seen: set[tuple[str, int, int, bool]] = set()
    out: list[EvidenceRef] = []
    for ref in refs:
        key = (ref.segment_id, ref.start_char, ref.end_char, ref.verified)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def _rule_fires(
    rule: dict, merged: dict[str, ItemResult], red_flags: list[RedFlag],
    checklist=None, ruled_items: frozenset[str] = frozenset(),
) -> tuple[bool, list[EvidenceRef]]:
    when = rule.get("when") or {}
    fired = False
    evidence: list[EvidenceRef] = []

    if "red_flag_severity" in when:
        severity = when["red_flag_severity"]
        matching = [f for f in red_flags if f.severity == severity]
        if matching:
            fired = True
            for f in matching:
                evidence.extend(f.evidence)

    if "item" in when and "status" in when:
        statuses = when["status"]
        if isinstance(statuses, str):
            statuses = [statuses]
        item = merged.get(when["item"])
        if item is not None and item.status.value in statuses:
            fired = True
            evidence.extend(item.evidence)

    if "other_item_status" in when:
        # Any Stage 2 item with no `item` rule of its own -- typically a
        # check added through a caller-supplied checklist. The red-flag list
        # item (C06) is skipped: the red_flag_severity rules cover it.
        statuses = when["other_item_status"]
        if isinstance(statuses, str):
            statuses = [statuses]
        for item_id, item in merged.items():
            cfg_item = config.get_item(item_id, checklist)
            if (cfg_item is None or cfg_item.stage != 2 or cfg_item.answer_type == "list"
                    or item_id in ruled_items):
                continue
            if item.status.value in statuses:
                fired = True
                evidence.extend(item.evidence)

    return fired, evidence


def _evaluate_rules(
    rules: list[dict], merged: dict[str, ItemResult], red_flags: list[RedFlag],
    checklist=None,
) -> tuple[bool, bool, list[str], list[str], list[EvidenceRef]]:
    escalate = False
    hold = False
    reasons: list[str] = []
    rule_ids: list[str] = []
    evidence: list[EvidenceRef] = []
    # Items that have their own `item` rule; `other_item_status` skips them.
    ruled_items = frozenset(
        (rule.get("when") or {}).get("item") for rule in rules if (rule.get("when") or {}).get("item")
    )

    for rule in rules:
        fired, rule_evidence = _rule_fires(rule, merged, red_flags, checklist, ruled_items)
        if not fired:
            continue
        escalate = escalate or bool(rule.get("escalate"))
        hold = hold or bool(rule.get("hold"))
        reasons.append(rule.get("reason", ""))
        rule_ids.append(rule.get("id", ""))
        evidence.extend(rule_evidence)

    return escalate, hold, reasons, rule_ids, _dedupe_refs(evidence)


def _stage2_summary(merged: dict[str, ItemResult], checklist=None) -> list[dict]:
    items = checklist if checklist is not None else config.load_checklist()
    order = {item.id: i for i, item in enumerate(items)}
    out = []
    for item_id, result in merged.items():
        cfg_item = config.get_item(item_id, checklist)
        if cfg_item is None or cfg_item.stage != 2:
            continue
        out.append({
            "id": item_id,
            "name": cfg_item.name,
            "status": result.status.value,
            "sub_criteria": [
                {"id": sc.id, "status": sc.status.value, "note": sc.note}
                for sc in result.sub_criteria
            ],
        })
    out.sort(key=lambda d: order.get(d["id"], 0))
    return out


def _red_flag_summary(red_flags: list[RedFlag]) -> list[dict]:
    return [
        {"category": f.category, "severity": f.severity, "description": f.description}
        for f in red_flags
    ]


def _has_unclear_stage2(merged: dict[str, ItemResult], checklist=None) -> bool:
    for item_id, result in merged.items():
        cfg_item = config.get_item(item_id, checklist)
        if cfg_item is None or cfg_item.stage != 2:
            continue
        if result.status in (Status.PARTIAL, Status.CANNOT_DETERMINE):
            return True
    return False


# ---------------------------------------------------------------------------
# Step 2: post-call actions.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _actions_output_model():
    """LLM output schema for `decide_actions`, built dynamically so
    `action_type` is constrained to exactly the keys in `ownership.yaml`
    (a bad action type then fails validation in `get_llm_response` instead
    of silently passing through). Cached: `ownership.yaml` doesn't change
    at runtime."""
    action_types = tuple(config.load_ownership().keys())
    proposed_action = create_model(
        "ProposedActionItem",
        action_type=(Literal[action_types], ...),
        description=(str, ...),
        evidence=(list[EvidenceQuote], Field(default_factory=list)),
    )
    return create_model(
        "ActionsOutput",
        actions=(list[proposed_action], Field(default_factory=list)),
    )


def _resolve_action(item, segments: list[dict], errors: list[str]) -> Action:
    refs, notes = resolve_evidence(segments, item.evidence)
    errors.extend(f"decide[actions]: {n}" for n in notes)
    if item.evidence and not refs:
        errors.append(
            f"decide[actions]: action {item.action_type!r} evidence did not "
            "resolve; kept without evidence"
        )
    return Action(
        action_type=item.action_type,
        description=item.description,
        owner_role=config.load_ownership().get(item.action_type),
        evidence=refs,
    )


# ---------------------------------------------------------------------------
# Node.
# ---------------------------------------------------------------------------

def decide_node(state: ComplianceState) -> dict:
    try:
        return _decide_node(state)
    except Exception as exc:  # noqa: BLE001 - never crash the graph
        return {
            "actions": [],
            "escalation": Escalation(source="undetermined").model_dump(),
            "item_results": [
                ItemResult(
                    item_id="C07", status=Status.CANNOT_DETERMINE,
                    sub_criteria=[
                        SubCriterionResult(id="no_escalation", status=Status.CANNOT_DETERMINE),
                        SubCriterionResult(id="no_hold", status=Status.CANNOT_DETERMINE),
                    ],
                    reasoning="Escalation decision failed.",
                ),
                ItemResult(
                    item_id="C08", status=Status.CANNOT_DETERMINE,
                    reasoning="Post-call action determination failed.",
                ),
                ItemResult(
                    item_id="C09", status=Status.NOT_APPLICABLE,
                    reasoning="No post-call actions to assign.",
                ),
            ],
            "errors": [f"decide: unexpected error: {type(exc).__name__}: {exc}"],
        }


def _decide_node(state: ComplianceState) -> dict:
    errors: list[str] = []
    merged = merged_results(state)
    red_flags: list[RedFlag] = list(
        state["verified_red_flags"] if "verified_red_flags" in state else state.get("red_flags", [])
    )
    segments = state.get("segments") or []
    call_intent = state.get("call_intent")
    quality_flags = state.get("quality_flags") or []

    # ---- 1. Escalation. --------------------------------------------------
    rules = _escalation_rules()
    checklist = config.active_checklist(state)
    escalate, hold, reasons, rule_ids, rule_evidence = _evaluate_rules(rules, merged, red_flags, checklist)

    if rule_ids:
        source = "rules"
    elif _has_unclear_stage2(merged, checklist):
        llm_result = get_llm_response(
            "decide_escalation", SYSTEM_ESCALATION,
            build_escalation_user(
                _stage2_summary(merged, checklist), _red_flag_summary(red_flags), quality_flags, call_intent,
            ),
            EscalationDecision,
        )
        if llm_result.ok:
            out = llm_result.value
            escalate, hold = out.escalate, out.hold
            reasons = [out.reason] if out.reason else []
            source = "llm"
        else:
            errors.append(f"decide[escalation]: {llm_result.error}")
            source = "undetermined"
    else:
        source = "none"

    escalation = Escalation(
        required=escalate, reasons=reasons, hold_transaction=hold,
        source=source, rule_ids=rule_ids,
    )

    # ---- 2. Post-call actions. --------------------------------------------
    actions: list[Action] = []
    c08_status = Status.PASS
    if segments:
        llm_result = get_llm_response(
            "decide_actions", SYSTEM_ACTIONS,
            build_actions_user(
                format_transcript(segments), call_intent, escalation.model_dump(),
                _stage2_summary(merged, checklist),
            ),
            _actions_output_model(),
        )
        if llm_result.ok:
            for item in llm_result.value.actions:
                actions.append(_resolve_action(item, segments, errors))
        else:
            errors.append(f"decide[actions]: {llm_result.error}")
            c08_status = Status.CANNOT_DETERMINE

    if escalation.required and not any(a.action_type == "raise_case" for a in actions):
        actions.append(Action(
            action_type="raise_case",
            description=(
                f"Raise a compliance case: {'; '.join(reasons)}" if reasons
                else "Raise a compliance case."
            ),
            owner_role=config.load_ownership().get("raise_case"),
            evidence=[],
        ))

    # ---- 3. Ownership is already filled in `_resolve_action` / above. -----

    # ---- Stage 3 ItemResults. ---------------------------------------------
    if source == "undetermined":
        c07_subs = [
            SubCriterionResult(id="no_escalation", status=Status.CANNOT_DETERMINE, note="Escalation could not be determined."),
            SubCriterionResult(id="no_hold", status=Status.CANNOT_DETERMINE, note="Hold could not be determined."),
        ]
    else:
        c07_subs = [
            SubCriterionResult(
                id="no_escalation",
                status=Status.FAIL if escalate else Status.PASS,
                evidence=rule_evidence if escalate else [],
                note="; ".join(r for r in reasons if r) if escalate else "No escalation required.",
            ),
            SubCriterionResult(
                id="no_hold",
                status=Status.FAIL if hold else Status.PASS,
                evidence=rule_evidence if hold else [],
                note="; ".join(r for r in reasons if r) if hold else "No hold required.",
            ),
        ]
    c07_status = rollup_status(c07_subs)
    source_desc = source + (f" {', '.join(rule_ids)}" if rule_ids else "")
    c07 = ItemResult(
        item_id="C07",
        status=c07_status,
        confidence=1.0 if source in ("rules", "none") else (0.5 if source == "llm" else 0.0),
        sub_criteria=c07_subs,
        evidence=_dedupe_refs([ref for sc in c07_subs for ref in sc.evidence]),
        reasoning=f"Escalation: {'required' if escalate else 'not required'} (source: {source_desc}).",
    )

    c08_evidence = _dedupe_refs([ref for a in actions for ref in a.evidence])
    if c08_status == Status.CANNOT_DETERMINE:
        c08_reasoning = "Post-call action determination failed."
    elif actions:
        c08_reasoning = f"{len(actions)} follow-up action(s) identified."
    else:
        c08_reasoning = "No follow-up actions required."
    c08 = ItemResult(
        item_id="C08",
        status=c08_status,
        confidence=1.0 if c08_status == Status.PASS else 0.0,
        evidence=c08_evidence,
        reasoning=c08_reasoning,
    )

    if not actions:
        c09 = ItemResult(
            item_id="C09", status=Status.NOT_APPLICABLE, confidence=1.0,
            reasoning="No post-call actions to assign.",
        )
    else:
        c09 = ItemResult(
            item_id="C09", status=Status.PASS, confidence=1.0,
            reasoning="; ".join(f"{a.action_type} -> {a.owner_role}" for a in actions),
        )

    out: dict = {
        "actions": actions,
        "escalation": escalation.model_dump(),
        "item_results": [c07, c08, c09],
    }
    if errors:
        out["errors"] = errors
    return out
