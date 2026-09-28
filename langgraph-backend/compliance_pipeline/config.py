"""Load and validate the YAML config in `compliance_pipeline/config/` (cached).

Three files: `checklist.yaml` (the default checklist), `ownership.yaml`
(action_type -> owning role), `escalation_rules.yaml` (deterministic
escalation rules). All are loaded once per process via `lru_cache`.

`checklist.yaml` is only the *default*: a caller may send its own
`checklist` with the request (e.g. a human edited it before the API call).
`resolve_checklist` turns that into the checklist for one run, `ingest`
stores the result in state, and every node reads it back with
`active_checklist(state)`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import ValidationError

from compliance_pipeline.schemas import ChecklistItem

CONFIG_DIR = Path(__file__).resolve().parent / "config"

# The only `applies_when` condition keys `gate.py` knows how to evaluate.
KNOWN_APPLIES_WHEN_KEYS = {"always", "call_intent"}

# Sub-criterion id given to a caller item that has no `sub_criteria` of its
# own: the item's question (or `pass_condition`) becomes its one criterion.
DEFAULT_SUB_CRITERION_ID = "met"


def _validate_applies_when(item_id: str, applies_when: dict) -> None:
    unknown = set(applies_when) - KNOWN_APPLIES_WHEN_KEYS
    if unknown:
        raise ValueError(
            f"checklist item {item_id!r} has unknown applies_when key(s): "
            f"{sorted(unknown)}; expected one of {sorted(KNOWN_APPLIES_WHEN_KEYS)}"
        )


@lru_cache(maxsize=1)
def load_checklist() -> list[ChecklistItem]:
    """The default checklist items (`checklist.yaml`), in config order."""
    raw = yaml.safe_load((CONFIG_DIR / "checklist.yaml").read_text(encoding="utf-8")) or []
    items = [ChecklistItem.model_validate(entry) for entry in raw]
    for item in items:
        _validate_applies_when(item.id, item.applies_when)
    return items


@lru_cache(maxsize=1)
def load_ownership() -> dict[str, str]:
    """action_type -> owning role, e.g. {"place_order": "Operations"}."""
    raw = yaml.safe_load((CONFIG_DIR / "ownership.yaml").read_text(encoding="utf-8")) or {}
    return dict(raw)


@lru_cache(maxsize=1)
def load_escalation_rules() -> list[dict]:
    """Deterministic escalation rules, evaluated in order by `decide`."""
    raw = yaml.safe_load((CONFIG_DIR / "escalation_rules.yaml").read_text(encoding="utf-8")) or []
    return list(raw)


def get_item(item_id: str, checklist: list[ChecklistItem] | None = None) -> ChecklistItem | None:
    """The item with this id in `checklist` (default: `checklist.yaml`), or None."""
    for item in checklist if checklist is not None else load_checklist():
        if item.id == item_id:
            return item
    return None


# ---------------------------------------------------------------------------
# Per-run checklist: the caller's checklist, or the default.
# ---------------------------------------------------------------------------

def _new_item_defaults(entry: dict) -> dict:
    """Defaults for a caller item whose id is not in `checklist.yaml`. Only
    `id` and `question` are required: it becomes a Stage 2 `criteria` check
    with one sub-criterion, and its question doubles as the rule query."""
    question = str(entry.get("question") or "").strip()
    return {
        "name": entry.get("id"),
        "stage": 2,
        "answer_type": "criteria",
        "applies_when": {"always": True},
        "rule_queries": [question] if question else [],
        "sub_criteria": [{
            "id": DEFAULT_SUB_CRITERION_ID,
            "text": str(entry.get("pass_condition") or question).strip(),
        }],
    }


def _resolve_item(index: int, entry, defaults_by_id: dict[str, ChecklistItem]) -> ChecklistItem:
    where = f"checklist[{index}]"
    if not isinstance(entry, dict):
        raise ValueError(f"{where} must be an object with at least 'id' and 'question'")
    item_id = str(entry.get("id") or "").strip()
    if not item_id:
        raise ValueError(f"{where} is missing 'id'")
    where = f"{where} ({item_id!r})"

    # Unknown keys (e.g. old-format `question_type`) are ignored.
    given = {k: v for k, v in entry.items() if k in ChecklistItem.model_fields}
    given["id"] = item_id
    base = defaults_by_id.get(item_id)
    if base is not None:
        # Editing a default item: fields left out keep their default values.
        data = {**base.model_dump(), **given}
    else:
        if not str(entry.get("question") or "").strip():
            raise ValueError(f"{where} is a new item and needs a 'question'")
        data = {**_new_item_defaults(entry), **given}

    try:
        item = ChecklistItem.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"{where} is invalid: {exc}") from exc

    _validate_applies_when(item_id, item.applies_when)
    if base is not None:
        if item.stage != base.stage or item.answer_type != base.answer_type:
            raise ValueError(
                f"{where} cannot change 'stage' or 'answer_type' of a default item "
                f"(stage {base.stage}, answer_type {base.answer_type!r})"
            )
    elif item.stage != 2 or item.answer_type != "criteria":
        raise ValueError(
            f"{where}: new items must be Stage 2 'criteria' checks; Stage 1/3 items "
            "and the red-flag list item are computed by the pipeline itself"
        )
    if item.stage == 2:
        if not item.question.strip():
            raise ValueError(f"{where} needs a non-empty 'question'")
        if not item.sub_criteria:
            raise ValueError(f"{where} needs at least one sub-criterion")
        sub_ids = [sc.id for sc in item.sub_criteria]
        if len(set(sub_ids)) != len(sub_ids):
            raise ValueError(f"{where} has duplicate sub-criterion ids")
    return item


def resolve_checklist(raw: list | None) -> list[ChecklistItem]:
    """The checklist for one run.

    - No checklist (None / empty list) -> the default `checklist.yaml`.
    - Otherwise the caller's Stage 2 items replace the default Stage 2
      items, in the caller's order: leaving a default item out skips it, a
      new id adds a check. An item that reuses a default id (e.g. `C01`)
      only needs the fields it changes.
    - Stage 1 (C02 call intent) and Stage 3 (C07-C09) items are computed by
      the pipeline, so they are always kept; a caller may send them to edit
      their text, but cannot add new ones.

    Raises `ValueError` with a readable message on an invalid checklist.
    """
    default = load_checklist()
    if raw is None or (isinstance(raw, list) and not raw):
        return list(default)
    if not isinstance(raw, list):
        raise ValueError("'checklist' must be a list of checklist items")

    defaults_by_id = {item.id: item for item in default}
    given: list[ChecklistItem] = []
    seen: set[str] = set()
    for index, entry in enumerate(raw):
        item = _resolve_item(index, entry, defaults_by_id)
        if item.id in seen:
            raise ValueError(f"checklist has duplicate id {item.id!r}")
        seen.add(item.id)
        given.append(item)

    given_by_id = {item.id: item for item in given}
    stage2 = [item for item in given if item.stage == 2]
    if not stage2:
        raise ValueError("checklist has no Stage 2 checks to run")
    system = {stage: [given_by_id.get(item.id, item) for item in default if item.stage == stage]
              for stage in (1, 3)}
    return system[1] + stage2 + system[3]


def active_checklist(state) -> list[ChecklistItem]:
    """The checklist for this run, as `ingest` stored it in state; the
    default when state has none (e.g. a node run on its own in a test)."""
    raw = (state or {}).get("checklist")
    if not raw:
        return load_checklist()
    return [item if isinstance(item, ChecklistItem) else ChecklistItem.model_validate(item) for item in raw]
