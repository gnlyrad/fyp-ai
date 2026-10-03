import { AlertTriangle, GitCompare } from "lucide-react";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { DataTable, type Column } from "../components/DataTable";
import { Page } from "../components/Page";
import { Badge, Card, Disclosure, Empty, ErrorPanel, Loading, MetricLabel, Tip, cx } from "../components/ui";
import { useDataset, type RunSummary } from "../lib/api";
import { fraction, plural, ratioPct, sgt, shortHash } from "../lib/format";
import { useMetaApi } from "../lib/meta";
import { paths, useSource } from "../lib/url";

const HISTORY_METRICS = [
  "compliance.decision_coverage",
  "compliance.automatic_breach_detection_rate",
  "compliance.false_clearance_rate",
  "compliance.prediction_failure_rate",
  "extraction.evidence_coverage",
];

export function RunStatusBadge({ run }: { run: RunSummary }) {
  const status = run.status;
  const tone = status.tone === "good" ? "good" : status.tone === "bad" ? "bad" : "warn";
  const detail = status.detail || (status.raw === "completed_with_errors" && status.prediction_failures === 0
    ? "pipeline warnings only" : "");
  return (
    <Tip content={{ title: status.label, what: status.description, important: detail || undefined }}>
      <span tabIndex={0} className="focus-ring inline-flex rounded-full">
        <Badge tone={tone}>{status.label}{status.prediction_failures > 0 && ` · ${plural(status.prediction_failures, "failure")}`}</Badge>
      </span>
    </Tip>
  );
}

export default function DatasetPage() {
  const source = useSource();
  const { datasetId = "" } = useParams();
  const navigate = useNavigate();
  const { wording } = useMetaApi();
  const query = useDataset(source, datasetId);
  const [selected, setSelected] = useState<string[]>([]);

  if (query.isPending) return <Page crumbs={[{ label: "Dataset" }]}><Loading /></Page>;
  if (query.isError) return <Page crumbs={[{ label: "Dataset" }]}><ErrorPanel error={query.error} /></Page>;
  const { dataset, runs } = query.data;

  const toggle = (runId: string) =>
    setSelected((current) => current.includes(runId) ? current.filter((id) => id !== runId) : [...current, runId].slice(-2));
  const compare = () => {
    const [a, b] = selected;
    navigate(paths.compare(source, datasetId, a, b));
  };

  const columns: Column<RunSummary>[] = [
    {
      key: "select", header: <span className="sr-only">Select for comparison</span>,
      cell: (run) => (
        <input type="checkbox" aria-label={`Select run ${run.run_id} for comparison`} checked={selected.includes(run.run_id)}
          onClick={(event) => event.stopPropagation()} onChange={() => toggle(run.run_id)}
          className="h-3.5 w-3.5 cursor-pointer accent-[var(--accent)]" />
      ),
    },
    {
      key: "evaluated", header: "Evaluated (SGT)", sort: (run) => run.evaluated_at ?? "",
      cell: (run) => (
        <div>
          <div className="num whitespace-nowrap text-ink">{sgt(run.evaluated_at)}</div>
          <div className="font-mono text-[11px] text-ink-3">{run.run_id}</div>
        </div>
      ),
    },
    {
      key: "status", header: "Status", sort: (run) => run.status.label,
      cell: (run) => (
        <div className="flex flex-wrap items-center gap-1">
          <RunStatusBadge run={run} />
          {!run.is_complete && <Badge tone="neutral" icon={false}>Partial / debug</Badge>}
        </div>
      ),
    },
    ...HISTORY_METRICS.map<Column<RunSummary>>((id) => ({
      key: id, align: "right",
      header: <MetricLabel id={id} />,
      sort: (run) => run.metrics[id]?.percentage ?? null,
      cell: (run) => (
        <span title={fraction(run.metrics[id])} className="num text-ink">{ratioPct(run.metrics[id])}</span>
      ),
    })),
    {
      key: "commit", header: "Commit", sort: (run) => run.graph_commit ?? "",
      cell: (run) => (
        <span className="inline-flex items-center gap-1 font-mono text-xs text-ink-2">
          {shortHash(run.graph_commit)}
          {run.graph_worktree_dirty && (
            <Tip content={{ title: "Uncommitted changes", what: "Uncommitted code changes were present when this evaluation ran, so the commit does not fully identify the code." }}>
              <span tabIndex={0} className="focus-ring rounded text-ink-warn"><AlertTriangle size={12} /></span>
            </Tip>
          )}
        </span>
      ),
    },
  ];

  return (
    <Page crumbs={[{ label: "Datasets", to: paths.datasets(source) }, { label: dataset.label }]}>
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-ink">{dataset.label}</h1>
          <p className="mt-0.5 text-[13px] text-ink-2">
            {plural(dataset.call_count, "call")} · {plural(dataset.question_count, "question")}
            {" "}({dataset.compliance_questions} compliance, {dataset.extraction_questions} extraction) · {plural(runs.length, "run")}
          </p>
        </div>
        <Disclosure title="Dataset lineage">
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-[11px] text-ink-2">
            <dt className="text-ink-3">dataset</dt><dd>{dataset.dataset_sha256}</dd>
            <dt className="text-ink-3">checklist</dt><dd>{dataset.checklist_sha256 ?? "—"}</dd>
            <dt className="text-ink-3">manifest</dt><dd>{dataset.manifest_sha256 ?? "—"}</dd>
            <dt className="text-ink-3">format</dt><dd>{dataset.format}</dd>
            <dt className="text-ink-3">calls</dt><dd>{dataset.call_ids.join(", ")}</dd>
          </dl>
        </Disclosure>
      </div>

      <Card>
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-3 py-2">
          <h2 className="text-sm font-semibold text-ink">Run history</h2>
          <div className="flex items-center gap-2 text-xs text-ink-3">
            <span>{selected.length === 2 ? "2 runs selected" : "Tick two runs to compare them"}</span>
            <button type="button" onClick={compare} disabled={selected.length !== 2}
              className={cx("focus-ring inline-flex items-center gap-1 rounded-md px-2.5 py-1 font-medium",
                selected.length === 2 ? "bg-accent text-white" : "cursor-not-allowed border border-line text-ink-3")}>
              <GitCompare size={13} /> Compare selected
            </button>
          </div>
        </div>
        <DataTable id="runs" columns={columns} rows={runs} rowKey={(run) => run.run_id}
          rowTitle="Open this run"
          onRowClick={(run) => navigate(paths.run(source, datasetId, run.run_id))}
          empty={<Empty>No evaluation runs belong to this dataset yet.</Empty>} />
      </Card>
      <p className="mt-2 text-xs text-ink-3">
        {wording("compliance.automatic_breach_detection_rate").label} counts every actual breach, including those left UNCLEAR or
        lost to failures. Breach recall is shown inside each run.
      </p>
    </Page>
  );
}
