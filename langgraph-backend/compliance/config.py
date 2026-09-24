"""Load and validate the YAML config in `compliance/config/` (cached).

Three files: `checklist.yaml` (the checklist itself), `ownership.yaml`
(action_type -> owning role), `escalation_rules.yaml` (deterministic
escalation rules). All are loaded once per process via `lru_cache`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from compliance.schemas import ChecklistItem

CONFIG_DIR = Path(__file__).resolve().parent / "config"

# The only `applies_when` condition keys `gate.py` knows how to evaluate.
KNOWN_APPLIES_WHEN_KEYS = {"always", "call_intent", "has_orders", "product_discussed"}


def _validate_applies_when(item_id: str, applies_when: dict) -> None:
    unknown = set(applies_when) - KNOWN_APPLIES_WHEN_KEYS
    if unknown:
        raise ValueError(
            f"checklist.yaml item {item_id!r} has unknown applies_when key(s): "
            f"{sorted(unknown)}; expected one of {sorted(KNOWN_APPLIES_WHEN_KEYS)}"
        )


@lru_cache(maxsize=1)
def load_checklist() -> list[ChecklistItem]:
    """All checklist items, in config order."""
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


def get_item(item_id: str) -> ChecklistItem | None:
    """The checklist item with this id, or None if it does not exist."""
    for item in load_checklist():
        if item.id == item_id:
            return item
    return None
