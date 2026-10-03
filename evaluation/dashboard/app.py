"""Shared evaluation dashboard backed by local artifacts or Supabase."""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EVALUATION_ROOT.parent
if str(EVALUATION_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALUATION_ROOT))

from dashboard.domain import DatasetRecord, ResultRecord, RunRecord  # noqa: E402
from dashboard.local_repository import LocalEvaluationRepository  # noqa: E402
from dashboard.repository import EvaluationRepository, RepositoryError  # noqa: E402
from dashboard.view_models import (  # noqa: E402
    compare_metric_rows,
    comparison_incompatibilities,
    display_timestamp,
    format_percentage,
    metric_help,
    metric_index,
    metric_percentage,
    nested,
)

load_dotenv(REPOSITORY_ROOT / ".env")
load_dotenv(EVALUATION_ROOT / ".env")

st.set_page_config(page_title="FYP Evaluation Dashboard", page_icon="📊", layout="wide")

COMPLIANCE_CAPABILITY = [
    "compliance.decision_coverage",
    "compliance.abstention_rate",
    "compliance.prediction_failure_rate",
]
COMPLIANCE_BINARY = [
    "compliance.breach_recall",
    "compliance.breach_precision",
    "compliance.binary_accuracy",
]
COMPLIANCE_BREACH = [
    "compliance.automatic_breach_detection_rate",
    "compliance.false_clearance_rate",
    "compliance.gold_breach_abstention_rate",
    "compliance.gold_breach_prediction_failure_rate",
]
EXTRACTION_HEADLINE = ["extraction.evidence_coverage", "extraction.prediction_failure_rate"]
EVIDENCE_METRICS = [
    "compliance.evidence_precision",
    "compliance.evidence_recall",
    "compliance.evidence_exact_set_match",
    "compliance.invalid_evidence_reference_rate",
    "extraction.evidence_precision",
    "extraction.evidence_recall",
    "extraction.evidence_exact_set_match",
    "extraction.invalid_evidence_reference_rate",
]


def navigate(page: str, **values: Any) -> None:
    st.session_state.page = page
    for key, value in values.items():
        st.session_state[key] = value
    st.rerun()


def metric_card(node: dict[str, Any], definitions: dict[str, dict[str, Any]], metric_id: str) -> None:
    definition = definitions[metric_id]
    st.metric(
        definition["label"],
        format_percentage(metric_percentage(node, definition)),
        help=metric_help(definition),
    )


def metric_row(node: dict[str, Any], definitions: dict[str, dict[str, Any]], metric_ids: list[str]) -> None:
    columns = st.columns(len(metric_ids))
    for column, metric_id in zip(columns, metric_ids):
        with column:
            metric_card(node, definitions, metric_id)


def dataset_label(dataset: DatasetRecord) -> str:
    return dataset.name.replace("_", " ").title()


def source_repository(source: str) -> EvaluationRepository:
    if source == "Local":
        return LocalEvaluationRepository(EVALUATION_ROOT)
    from dashboard.supabase_repository import SupabaseEvaluationRepository  # noqa: PLC0415
    return SupabaseEvaluationRepository.from_environment()


def breadcrumb(items: list[tuple[str, str, dict[str, Any]]]) -> None:
    columns = st.columns([1] * len(items))
    for column, (label, page, values) in zip(columns, items):
        with column:
            if st.button(label, key=f"crumb-{page}-{label}", width="stretch"):
                navigate(page, **values)


def run_table_row(run: RunRecord, definitions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    overall = run.summary.get("overall", {})
    return {
        "Evaluated At": display_timestamp(run.evaluated_at),
        "Run ID": run.run_id,
        "Status": run.status,
        "Scope": "Complete" if run.is_complete else "Partial / debug",
        "Git Commit": (run.metadata.get("graph_commit") or "")[:8],
        "Decision Coverage": format_percentage(metric_percentage(overall, definitions["compliance.decision_coverage"])),
        "Breach Detection": format_percentage(metric_percentage(overall, definitions["compliance.automatic_breach_detection_rate"])),
        "Extraction Coverage": format_percentage(metric_percentage(overall, definitions["extraction.evidence_coverage"])),
    }


def render_dataset_browser(repository: EvaluationRepository, datasets: list[DatasetRecord]) -> None:
    st.title("Evaluation datasets")
    st.caption(f"Source: {repository.source_name}. Select an immutable benchmark snapshot.")
    if not datasets:
        st.info("No datasets are available in this source.")
        return
    rows = []
    for dataset in datasets:
        runs = repository.list_runs(dataset.dataset_id)
        rows.append({
            "Dataset": dataset_label(dataset),
            "Folder / key": dataset.name,
            "Calls": dataset.call_count,
            "Questions": dataset.question_count,
            "Runs": len(runs),
            "Dataset hash": dataset.dataset_sha256[:12],
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    selected = st.selectbox("Dataset", datasets, format_func=dataset_label)
    if st.button("Open dataset", type="primary"):
        navigate("dataset", dataset_id=selected.dataset_id)


def render_dataset(repository: EvaluationRepository, dataset: DatasetRecord, definitions: dict[str, dict[str, Any]]) -> None:
    breadcrumb([("All datasets", "datasets", {})])
    st.title(dataset_label(dataset))
    columns = st.columns(4)
    columns[0].metric("Calls", dataset.call_count)
    columns[1].metric("Questions", dataset.question_count)
    columns[2].metric("Runs", len(repository.list_runs(dataset.dataset_id)))
    columns[3].metric("Source", repository.source_name)
    with st.expander("Dataset lineage"):
        st.code(json.dumps({
            "dataset_sha256": dataset.dataset_sha256,
            "checklist_sha256": dataset.checklist_sha256,
            "manifest_sha256": dataset.manifest_sha256,
            "format": dataset.dataset_format,
            "call_ids": dataset.call_ids,
        }, indent=2))
    runs = repository.list_runs(dataset.dataset_id)
    st.subheader("Run history")
    if not runs:
        st.info("No evaluation runs belong to this dataset yet.")
        return
    st.dataframe([run_table_row(run, definitions) for run in runs], width="stretch", hide_index=True)
    selected = st.selectbox("Run", runs, format_func=lambda run: f"{display_timestamp(run.evaluated_at)} — {run.run_id}")
    left, right = st.columns(2)
    with left:
        if st.button("Open run", type="primary", width="stretch"):
            navigate("run", dataset_id=dataset.dataset_id, run_id=selected.run_id)
    with right:
        if st.button("Compare runs", width="stretch"):
            navigate("compare", dataset_id=dataset.dataset_id)


def render_run_header(dataset: DatasetRecord, run: RunRecord) -> None:
    breadcrumb([
        ("All datasets", "datasets", {}),
        (dataset_label(dataset), "dataset", {"dataset_id": dataset.dataset_id}),
    ])
    st.title(display_timestamp(run.evaluated_at))
    st.caption(f"{run.run_id} · {run.source} · {'Complete benchmark' if run.is_complete else 'Partial / debug run'}")
    if run.metadata.get("graph_worktree_dirty") is True:
        st.warning("This run used a dirty Git worktree. The recorded commit does not fully identify the executed source.")
    if run.status == "completed_with_errors":
        st.warning("Completed with errors. Prediction failures and diagnostics remain part of the evaluation.")
    elif run.status == "failed":
        st.error("Run failed.")


def render_overview(run: RunRecord, definitions: dict[str, dict[str, Any]]) -> None:
    overall = run.summary.get("overall", {})
    st.subheader("Family overview")
    compliance, extraction = st.columns(2)
    with compliance:
        st.markdown("#### Compliance Decisioning")
        for metric_id in ("compliance.decision_coverage", "compliance.breach_recall", "compliance.automatic_breach_detection_rate"):
            metric_card(overall, definitions, metric_id)
    with extraction:
        st.markdown("#### Structured Extraction")
        for metric_id in EXTRACTION_HEADLINE:
            metric_card(overall, definitions, metric_id)
    st.subheader("Where to investigate")
    st.markdown(
        "Use **By Call** for difficult scenarios, **By Question** for systematic weaknesses, "
        "and **Diagnostics** for operational failures."
    )


def render_compliance(run: RunRecord, definitions: dict[str, dict[str, Any]]) -> None:
    node = run.summary.get("overall", {})
    st.subheader("1. Decision Capability")
    st.caption("Population: all compliance questions.")
    metric_row(node, definitions, COMPLIANCE_CAPABILITY)
    st.subheader("2. Binary Decision Quality")
    st.caption("Population: completed PASS/FAIL decisions only.")
    metric_row(node, definitions, COMPLIANCE_BINARY)
    matrix = nested(node, "compliance.binary.confusion_matrix", {}) or {}
    st.markdown("##### Confusion matrix")
    st.dataframe([
        {"Gold / Prediction": "Gold FAIL", "Predicted FAIL": matrix.get("TP", 0), "Predicted PASS": matrix.get("FN", 0)},
        {"Gold / Prediction": "Gold PASS", "Predicted FAIL": matrix.get("FP", 0), "Predicted PASS": matrix.get("TN", 0)},
    ], hide_index=True, width="stretch")
    st.subheader("3. End-to-End Actual Breach Handling")
    st.caption("Population: all gold FAIL cases, including abstentions and prediction failures.")
    metric_row(node, definitions, COMPLIANCE_BREACH)
    counts = nested(node, "compliance.breach_outcomes.counts", {}) or {}
    st.markdown("##### Actual breach outcomes")
    st.dataframe([{
        "Detected": counts.get("detected", 0),
        "False clearance": counts.get("false_clearance", 0),
        "Abstained": counts.get("abstained", 0),
        "Prediction failure": counts.get("prediction_failure", 0),
        "Unexpected N/A": counts.get("unexpected_not_applicable", 0),
    }], hide_index=True, width="stretch")


def render_extraction(run: RunRecord, definitions: dict[str, dict[str, Any]], checklist: list[dict[str, Any]]) -> None:
    node = run.summary.get("overall", {})
    metric_row(node, definitions, EXTRACTION_HEADLINE)
    counts = nested(node, "extraction.counts", {}) or {}
    st.subheader("Population and outcomes")
    st.dataframe([
        {"Population": "All extraction questions", "Count": counts.get("total", 0)},
        {"Population": "PRESENT", "Count": counts.get("present_expected_values", 0)},
        {"Population": "NOT_STATED", "Count": counts.get("not_stated_expected_values", 0)},
        {"Population": "PRESENT → FOUND", "Count": counts.get("found", 0)},
        {"Population": "PRESENT → NOT_FOUND", "Count": counts.get("not_found", 0)},
        {"Population": "Prediction failure", "Count": counts.get("prediction_failures", 0)},
    ], hide_index=True, width="stretch")
    st.subheader("By extraction field")
    questions = {item["id"]: item for item in checklist if item.get("evaluation_family") == "extraction"}
    rows = []
    for question_id, summary in run.summary.get("by_question_id", {}).items():
        if question_id not in questions:
            continue
        rows.append({
            "Question": question_id,
            "Field": questions[question_id].get("field"),
            "Evidence Coverage": format_percentage(metric_percentage(summary, definitions["extraction.evidence_coverage"])),
            "Failure Rate": format_percentage(metric_percentage(summary, definitions["extraction.prediction_failure_rate"])),
            "Found / expected": f"{nested(summary, 'extraction.counts.found', 0)} / {nested(summary, 'extraction.counts.coverage_population', 0)}",
        })
    st.dataframe(rows, hide_index=True, width="stretch")


def call_summary_row(call_id: str, node: dict[str, Any], definitions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "Call": call_id,
        "Decision Coverage": format_percentage(metric_percentage(node, definitions["compliance.decision_coverage"])),
        "Breach Detection": format_percentage(metric_percentage(node, definitions["compliance.automatic_breach_detection_rate"])),
        "Failures": nested(node, "counts.execution_status.prediction_failure", 0),
        "Abstentions": nested(node, "compliance.outcome_population.abstentions", 0),
        "Extraction Coverage": format_percentage(metric_percentage(node, definitions["extraction.evidence_coverage"])),
    }


def render_by_call(repository: EvaluationRepository, run: RunRecord, definitions: dict[str, dict[str, Any]]) -> None:
    nodes = run.summary.get("by_call_id", {})
    rows = [call_summary_row(call_id, node, definitions) for call_id, node in nodes.items()]
    st.dataframe(rows, hide_index=True, width="stretch")
    if rows:
        selected = st.selectbox("Call", sorted(nodes))
        if st.button("View call", type="primary"):
            navigate("call", dataset_id=run.dataset_id, run_id=run.run_id, call_id=selected, return_page="run")


def question_summary_row(
    question_id: str,
    node: dict[str, Any],
    question: dict[str, Any],
    definitions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if question.get("evaluation_family") == "compliance":
        return {
            "Question": question_id,
            "Description": question.get("question"),
            "Decision Coverage": format_percentage(metric_percentage(node, definitions["compliance.decision_coverage"])),
            "Breach Recall": format_percentage(metric_percentage(node, definitions["compliance.breach_recall"])),
            "Failure Rate": format_percentage(metric_percentage(node, definitions["compliance.prediction_failure_rate"])),
            "Abstention": format_percentage(metric_percentage(node, definitions["compliance.abstention_rate"])),
        }
    return {
        "Question": question_id,
        "Description": question.get("question"),
        "Field": question.get("field"),
        "Evidence Coverage": format_percentage(metric_percentage(node, definitions["extraction.evidence_coverage"])),
        "Failure Rate": format_percentage(metric_percentage(node, definitions["extraction.prediction_failure_rate"])),
        "Found / expected": f"{nested(node, 'extraction.counts.found', 0)} / {nested(node, 'extraction.counts.coverage_population', 0)}",
    }


def render_by_question(
    repository: EvaluationRepository,
    run: RunRecord,
    definitions: dict[str, dict[str, Any]],
    checklist: list[dict[str, Any]],
) -> None:
    questions = {item["id"]: item for item in checklist}
    nodes = run.summary.get("by_question_id", {})
    family = st.radio("Family", ["compliance", "extraction"], horizontal=True, key="question-family")
    filtered = [item for item in checklist if item.get("evaluation_family") == family and item["id"] in nodes]
    rows = [question_summary_row(item["id"], nodes[item["id"]], item, definitions) for item in filtered]
    st.dataframe(rows, hide_index=True, width="stretch")
    if filtered:
        selected = st.selectbox("Question", [item["id"] for item in filtered], format_func=lambda qid: f"{qid} — {questions[qid]['question']}")
        if st.button("View question", type="primary"):
            navigate("question", dataset_id=run.dataset_id, run_id=run.run_id, question_id=selected, return_page="run")


def render_diagnostics(repository: EvaluationRepository, run: RunRecord, definitions: dict[str, dict[str, Any]]) -> None:
    st.subheader("Evidence diagnostics")
    metric_row(run.summary.get("overall", {}), definitions, EVIDENCE_METRICS[:4])
    metric_row(run.summary.get("overall", {}), definitions, EVIDENCE_METRICS[4:])
    st.subheader("Operational errors and pipeline diagnostics")
    errors = repository.get_errors(run.run_id)
    if errors:
        st.dataframe(errors, hide_index=True, width="stretch")
    else:
        st.success("No persisted errors or pipeline diagnostics.")


def render_run(repository: EvaluationRepository, dataset: DatasetRecord, run: RunRecord, definitions: dict[str, dict[str, Any]]) -> None:
    render_run_header(dataset, run)
    checklist = repository.get_checklist(dataset.dataset_id)
    tabs = st.tabs(
        ["Overview", "Compliance", "Structured Extraction", "By Call", "By Question", "Diagnostics", "Metadata"],
        key=f"run-tabs-{run.run_id}",
    )
    with tabs[0]:
        render_overview(run, definitions)
    with tabs[1]:
        render_compliance(run, definitions)
    with tabs[2]:
        render_extraction(run, definitions, checklist)
    with tabs[3]:
        render_by_call(repository, run, definitions)
    with tabs[4]:
        render_by_question(repository, run, definitions, checklist)
    with tabs[5]:
        render_diagnostics(repository, run, definitions)
    with tabs[6]:
        st.json(run.metadata, expanded=False)


def result_status(row: dict[str, Any]) -> str:
    if row.get("execution_status") == "prediction_failure":
        return f"Prediction failure: {row.get('failure_category') or 'unspecified'}"
    if row.get("evaluation_family") == "compliance":
        return str(row.get("compliance_outcome_status") or "")
    return str(row.get("extraction_evidence_status") or row.get("gold_value_status") or "")


def render_call(repository: EvaluationRepository, dataset: DatasetRecord, run: RunRecord, call_id: str, definitions: dict[str, dict[str, Any]]) -> None:
    breadcrumb([
        (dataset_label(dataset), "dataset", {"dataset_id": dataset.dataset_id}),
        (run.run_id, "run", {"dataset_id": dataset.dataset_id, "run_id": run.run_id}),
    ])
    st.title(call_id)
    node = run.summary.get("by_call_id", {}).get(call_id, {})
    st.dataframe([call_summary_row(call_id, node, definitions)], hide_index=True, width="stretch")
    rows = repository.list_results(run.run_id, call_id=call_id)
    st.subheader("Results in this call")
    st.dataframe([{
        "Question": row["question_id"],
        "Family": row["evaluation_family"],
        "Gold": row.get("gold_verdict") or row.get("expected_value") or row.get("gold_value_status"),
        "Prediction": row.get("predicted_verdict") or row.get("extraction_evidence_status"),
        "Status": result_status(row),
    } for row in rows], hide_index=True, width="stretch")
    selected = st.selectbox("Result", rows, format_func=lambda row: f"{row['question_id']} — {result_status(row)}")
    if st.button("View individual result", type="primary"):
        navigate("result", dataset_id=dataset.dataset_id, run_id=run.run_id, call_id=call_id,
                 question_id=selected["question_id"], return_page="call")


def render_question(repository: EvaluationRepository, dataset: DatasetRecord, run: RunRecord, question_id: str, definitions: dict[str, dict[str, Any]]) -> None:
    checklist = {item["id"]: item for item in repository.get_checklist(dataset.dataset_id)}
    question = checklist[question_id]
    breadcrumb([
        (dataset_label(dataset), "dataset", {"dataset_id": dataset.dataset_id}),
        (run.run_id, "run", {"dataset_id": dataset.dataset_id, "run_id": run.run_id}),
    ])
    st.title(f"{question_id} — {question['question']}")
    node = run.summary.get("by_question_id", {}).get(question_id, {})
    st.dataframe([question_summary_row(question_id, node, question, definitions)], hide_index=True, width="stretch")
    rows = repository.list_results(run.run_id, question_id=question_id)
    st.subheader("Results by call")
    st.dataframe([{
        "Call": row["call_id"],
        "Gold": row.get("gold_verdict") or row.get("expected_value") or row.get("gold_value_status"),
        "Prediction": row.get("predicted_verdict") or row.get("extraction_evidence_status"),
        "Status": result_status(row),
        "Evidence recall": row.get("evidence_recall"),
    } for row in rows], hide_index=True, width="stretch")
    selected = st.selectbox("Call result", rows, format_func=lambda row: f"{row['call_id']} — {result_status(row)}")
    if st.button("View individual result", type="primary"):
        navigate("result", dataset_id=dataset.dataset_id, run_id=run.run_id, call_id=selected["call_id"],
                 question_id=question_id, return_page="question")


def transcript_html(result: ResultRecord) -> str:
    gold = set(result.score.get("gold_evidence_turn_ids") or result.ground_truth.get("gold_evidence_turns") or [])
    predicted = set(result.score.get("predicted_evidence_turn_ids") or [])
    blocks = []
    for segment in result.transcript.get("audio_segments", []):
        turn_id = segment.get("id")
        if turn_id in gold and turn_id in predicted:
            background, label = "#d9f7e8", "Gold + predicted"
        elif turn_id in gold:
            background, label = "#fff0c2", "Gold only"
        elif turn_id in predicted:
            background, label = "#dcecff", "Predicted only"
        else:
            background, label = "transparent", ""
        role = result.speaker_roles.get(str(segment.get("speaker_label")), result.speaker_roles.get(segment.get("speaker_label"), segment.get("speaker_label")))
        badge = f" · {label}" if label else ""
        blocks.append(
            f'<div style="padding:.65rem .8rem;margin:.25rem 0;border-radius:.4rem;background:{background};">'
            f'<strong>Turn {html.escape(str(turn_id))} · {html.escape(str(role))}{html.escape(badge)}</strong><br>'
            f'{html.escape(str(segment.get("transcript") or ""))}</div>'
        )
    return "".join(blocks)


def render_result(repository: EvaluationRepository, dataset: DatasetRecord, run: RunRecord, call_id: str, question_id: str) -> None:
    result = repository.get_result(run.run_id, call_id, question_id)
    return_page = st.session_state.get("return_page", "run")
    return_values = {"dataset_id": dataset.dataset_id, "run_id": run.run_id}
    if return_page == "call":
        return_values["call_id"] = call_id
    elif return_page == "question":
        return_values["question_id"] = question_id
    breadcrumb([
        (dataset_label(dataset), "dataset", {"dataset_id": dataset.dataset_id}),
        (run.run_id, "run", {"dataset_id": dataset.dataset_id, "run_id": run.run_id}),
        ("Back to analysis", return_page, return_values),
    ])
    st.title(f"{call_id} × {question_id}")
    st.subheader(result.question.get("question", question_id))
    row = result.row
    gold, predicted = st.columns(2)
    with gold:
        st.markdown("#### Ground truth")
        st.write(row.get("gold_verdict") or row.get("expected_value") or row.get("gold_value_status") or "—")
        st.write("Gold evidence:", result.score.get("gold_evidence_turn_ids") or result.ground_truth.get("gold_evidence_turns") or [])
    with predicted:
        st.markdown("#### Pipeline result")
        st.write(row.get("predicted_verdict") or row.get("extraction_evidence_status") or row.get("native_verdict") or "—")
        st.write("Status:", result_status(row))
        st.write("Predicted evidence:", result.score.get("predicted_evidence_turn_ids") or [])
    evidence = result.score.get("evidence", {})
    metrics = st.columns(3)
    metrics[0].metric("Evidence precision", format_percentage(nested(evidence, "precision.percentage")))
    metrics[1].metric("Evidence recall", format_percentage(nested(evidence, "recall.percentage")))
    metrics[2].metric("Exact-set match", format_percentage(nested(evidence, "exact_set_match.percentage")))
    st.subheader("Transcript evidence")
    st.markdown(transcript_html(result), unsafe_allow_html=True)
    finding = nested(result.prediction, "graph_metadata.finding", {}) or {}
    reasoning = finding.get("reason") or finding.get("reasoning")
    if reasoning:
        st.subheader("Model reasoning")
        st.write(reasoning)
    with st.expander("Normalized prediction"):
        st.json(result.prediction, expanded=False)
    with st.expander("Score and diagnostics"):
        st.json(result.score, expanded=False)


def render_compare(repository: EvaluationRepository, dataset: DatasetRecord, definitions_raw: dict[str, Any]) -> None:
    breadcrumb([("Dataset", "dataset", {"dataset_id": dataset.dataset_id})])
    st.title("Compare compatible runs")
    runs = repository.list_runs(dataset.dataset_id)
    if len(runs) < 2:
        st.info("At least two runs are required.")
        return
    first = st.selectbox("Baseline", runs, format_func=lambda run: run.run_id, key="compare-first")
    candidates = [run for run in runs if run.run_id != first.run_id]
    second = st.selectbox("Candidate", candidates, format_func=lambda run: run.run_id, key="compare-second")
    reasons = comparison_incompatibilities(first, second)
    if reasons:
        st.error("These runs are not directly comparable.")
        for reason in reasons:
            st.write(f"- {reason}")
        return
    metric_ids = COMPLIANCE_CAPABILITY + COMPLIANCE_BINARY + COMPLIANCE_BREACH + EXTRACTION_HEADLINE
    st.dataframe(compare_metric_rows(first, second, definitions_raw, metric_ids), hide_index=True, width="stretch")
    st.caption("Deltas are neutral percentage-point differences. No unapproved performance threshold is applied.")


def main() -> None:
    st.sidebar.title("Evaluation Dashboard")
    default_source = os.getenv("EVALUATION_DEFAULT_SOURCE", "Shared").title()
    if default_source not in {"Shared", "Local"}:
        default_source = "Shared"
    source = st.sidebar.radio("Source", ["Shared", "Local"], index=0 if default_source == "Shared" else 1)
    if st.session_state.get("active_source") != source:
        st.session_state.active_source = source
        st.session_state.page = "datasets"
        for key in ("dataset_id", "run_id", "call_id", "question_id"):
            st.session_state.pop(key, None)
    try:
        repository = source_repository(source)
        definitions_raw = repository.get_metric_definitions()
        definitions = metric_index(definitions_raw)
        datasets = repository.list_datasets()
    except (RepositoryError, RuntimeError, ValueError) as error:
        st.error(str(error))
        if source == "Shared":
            st.info("Set SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY, or switch Source to Local.")
        return
    except Exception as error:
        if source != "Shared":
            raise
        st.error(f"Shared evaluation source is unavailable ({type(error).__name__}).")
        st.info("Check SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY, or switch Source to Local.")
        return
    page = st.session_state.get("page", "datasets")
    try:
        if page == "datasets":
            render_dataset_browser(repository, datasets)
            return
        dataset = repository.get_dataset(st.session_state.dataset_id)
        if page == "dataset":
            render_dataset(repository, dataset, definitions)
            return
        if page == "compare":
            render_compare(repository, dataset, definitions_raw)
            return
        run = repository.get_run(st.session_state.run_id)
        if page == "run":
            render_run(repository, dataset, run, definitions)
        elif page == "call":
            render_call(repository, dataset, run, st.session_state.call_id, definitions)
        elif page == "question":
            render_question(repository, dataset, run, st.session_state.question_id, definitions)
        elif page == "result":
            render_result(repository, dataset, run, st.session_state.call_id, st.session_state.question_id)
        else:
            navigate("datasets")
    except (RepositoryError, KeyError) as error:
        st.error(str(error))
        if st.button("Return to datasets"):
            navigate("datasets")
    except Exception as error:
        if source != "Shared":
            raise
        st.error(f"Shared evaluation source is unavailable ({type(error).__name__}).")
        if st.button("Return to datasets"):
            navigate("datasets")


if __name__ == "__main__":
    main()
