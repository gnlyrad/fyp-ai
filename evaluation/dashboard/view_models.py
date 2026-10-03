"""Pure presentation helpers shared by Streamlit views and tests."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from .domain import RunRecord


def nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def metric_index(definitions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in definitions.get("metrics", [])}


def metric_value(summary_node: dict[str, Any], definition: dict[str, Any]) -> Any:
    return nested(summary_node, definition["summary_path"])


def metric_percentage(summary_node: dict[str, Any], definition: dict[str, Any]) -> float | None:
    value = metric_value(summary_node, definition)
    if isinstance(value, dict):
        percentage = value.get("percentage")
        return float(percentage) if isinstance(percentage, (int, float)) else None
    return None


def metric_help(definition: dict[str, Any]) -> str:
    sections = [
        ("Meaning", definition.get("interpretation")),
        ("Formula", definition.get("formula")),
        ("Population", definition.get("population")),
        ("Watch out", definition.get("watch_out")),
    ]
    return "\n\n".join(f"**{label}:** {text}" for label, text in sections if text)


def format_percentage(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value - round(value)) < 1e-9:
        return f"{value:.0f}%"
    return f"{value:.1f}%"


def display_timestamp(value: str | None) -> str:
    if not value:
        return "Unknown"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.strftime("%d %b %Y · %H:%M:%S UTC")


def comparison_incompatibilities(first: RunRecord, second: RunRecord) -> list[str]:
    reasons: list[str] = []
    if not first.is_complete or not second.is_complete:
        reasons.append("Both runs must cover the complete dataset call set.")
    comparisons = (
        ("dataset snapshot", ("dataset_snapshot", "dataset_sha256")),
        ("checklist", ("dataset_snapshot", "checklist_sha256")),
        ("scorer version", (None, "scorer_version")),
        ("metric-definition version", (None, "metric_definitions_version")),
        ("metric-definition content", (None, "metric_definitions_sha256")),
        ("artifact schema", (None, "artifact_schema_version")),
    )
    for label, (parent, key) in comparisons:
        left = first.metadata.get(parent, {}).get(key) if parent else first.metadata.get(key)
        right = second.metadata.get(parent, {}).get(key) if parent else second.metadata.get(key)
        if not left or not right:
            reasons.append(f"Missing {label} fingerprint/version.")
        elif left != right:
            reasons.append(f"Different {label}.")
    return reasons


def compare_metric_rows(
    first: RunRecord,
    second: RunRecord,
    definitions: dict[str, Any],
    metric_ids: list[str],
) -> list[dict[str, Any]]:
    index = metric_index(definitions)
    left = first.summary.get("overall", {})
    right = second.summary.get("overall", {})
    rows = []
    for metric_id in metric_ids:
        definition = index[metric_id]
        first_value = metric_percentage(left, definition)
        second_value = metric_percentage(right, definition)
        rows.append({
            "Metric": definition["label"],
            first.run_id: format_percentage(first_value),
            second.run_id: format_percentage(second_value),
            "Delta (percentage points)": (
                None if first_value is None or second_value is None else round(second_value - first_value, 2)
            ),
        })
    return rows
