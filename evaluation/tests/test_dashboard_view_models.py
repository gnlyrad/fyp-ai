from dashboard.domain import RunRecord
from dashboard.view_models import comparison_incompatibilities, metric_help, metric_percentage


def _run(run_id="run-a", complete=True, **overrides):
    metadata = {
        "dataset_snapshot": {"dataset_sha256": "dataset", "checklist_sha256": "checklist"},
        "scorer_version": "scorer",
        "metric_definitions_version": "2",
        "metric_definitions_sha256": "definitions",
        "artifact_schema_version": "2",
        **overrides,
    }
    return RunRecord(run_id, "dataset", "Local", metadata, {"overall": {}}, complete)


def test_metric_helpers_use_definition_path_and_text():
    definition = {
        "summary_path": "compliance.coverage",
        "interpretation": "Meaning",
        "formula": "A / B",
        "population": "All cases",
        "watch_out": "Caution",
    }
    node = {"compliance": {"coverage": {"percentage": 75.0}}}
    assert metric_percentage(node, definition) == 75.0
    help_text = metric_help(definition)
    assert all(value in help_text for value in ("Meaning", "A / B", "All cases", "Caution"))


def test_comparison_requires_all_locked_compatibility_fields():
    assert comparison_incompatibilities(_run(), _run("run-b")) == []
    reasons = comparison_incompatibilities(_run(), _run("run-b", scorer_version="other"))
    assert reasons == ["Different scorer version."]
    assert "Both runs must cover" in comparison_incompatibilities(_run(complete=False), _run("run-b"))[0]


def test_comparison_rejects_missing_fingerprint():
    reasons = comparison_incompatibilities(_run(metric_definitions_sha256=None), _run("run-b"))
    assert "Missing metric-definition content fingerprint/version." in reasons
