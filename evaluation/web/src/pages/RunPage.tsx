import { AlertTriangle } from "lucide-react";
import { useParams } from "react-router-dom";
import { Page } from "../components/Page";
import { Badge, Disclosure, ErrorPanel, Json, Loading, Tip, cx } from "../components/ui";
import { useRun, type RunDetail } from "../lib/api";
import { duration, money, plural, sgt, shortHash, tokens } from "../lib/format";
import { paths, useQueryState, useSource } from "../lib/url";
import Analysis from "./run/Analysis";
import { RunProvider } from "./run/context";
import Diagnostics from "./run/Diagnostics";
import Overview from "./run/Overview";
import { PanelHost } from "./run/Panels";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "analysis", label: "Analysis" },
  { id: "diagnostics", label: "Diagnostics" },
] as const;

function RunDetails({ run }: { run: RunDetail["run"] }) {
  const m = run.metadata;
  const model = m.model_configuration ?? {};
  const snapshot = m.dataset_snapshot ?? {};
  const items: [string, string][] = [
    ["Run ID", run.run_id],
    ["Run kind", m.run_kind ?? "—"],
    ["Started", sgt(m.started_at)],
    ["Completed", sgt(m.completed_at)],
    ["Total run time", duration(run.operations.total_latency_ms)],
    ["Average per call", duration(run.operations.avg_call_latency_ms)],
    ["Tokens used", run.operations.tokens
      ? `${tokens(run.operations.tokens.total_tokens)} (${tokens(run.operations.tokens.input_tokens)} in, ${tokens(run.operations.tokens.output_tokens)} out)`
      : "Not recorded for this run"],
    ...Object.entries(run.operations.cost?.prices ?? {}).map(([model, price]): [string, string] => [
      `Price used: ${model}`,
      `${money(price.input, run.operations.cost!.currency)} in / ${money(price.output, run.operations.cost!.currency)} out per 1M tokens`
        + `${price.cached_input != null ? ` (cached ${money(price.cached_input, run.operations.cost!.currency)})` : ""}`
        + `${price.source ? ` · ${price.source.replace(/^https?:\/\//, "")}` : ""}${price.as_of ? ` · ${price.as_of}` : ""}`,
    ]),
    ["Estimated cost", !run.operations.cost ? "Not recorded for this run"
      : run.operations.cost.amount === null ? `No price set for ${run.operations.cost.unpriced_models.join(", ")}`
      : `${money(run.operations.cost.amount, run.operations.cost.currency)} (${money(run.operations.cost.per_call, run.operations.cost.currency)} per call)`],
    ["Git commit", m.graph_commit ?? "—"],
    ["Git author", m.git_commit_author ?? "—"],
    ["Commit time", sgt(m.git_commit_timestamp)],
    ["Chat model", model.chat_model_id ?? "—"],
    ["Embedding model", model.embedding_model_id ?? "—"],
    ["Temperature", String(model.temperature ?? "—")],
    ["Context mode", m.execution_configuration?.context_mode ?? "—"],
    ["Scorer", m.scorer_version ?? "—"],
    ["Metric definitions", `${m.metric_definitions_version ?? "—"} · ${shortHash(m.metric_definitions_sha256, 12)}`],
    ["Artifact schema", m.artifact_schema_version ?? "—"],
    ["Dataset snapshot", shortHash(snapshot.dataset_sha256, 12)],
    ["Checklist", shortHash(snapshot.checklist_sha256, 12)],
    ["Calls evaluated", (m.execution_configuration?.selected_call_ids ?? snapshot.call_ids ?? []).join(", ")],
  ];
  if (m.derived_from) items.splice(2, 0, ["Rescored from", String(m.derived_from.run_id ?? JSON.stringify(m.derived_from))]);
  return (
    <div className="rounded-md border border-line bg-surface p-3">
      <dl className="grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2 lg:grid-cols-3">
        {items.map(([label, value]) => (
          <div key={label} className="flex min-w-0 gap-2">
            <dt className="w-32 shrink-0 text-ink-3">{label}</dt>
            <dd className="truncate font-mono text-ink-2" title={value}>{value}</dd>
          </div>
        ))}
      </dl>
      <Disclosure title="Full metadata (JSON)" className="mt-3"><Json value={m} /></Disclosure>
    </div>
  );
}

function RunHeader({ data }: { data: RunDetail }) {
  const { run } = data;
  const tone = run.status.tone === "good" ? "good" : run.status.tone === "bad" ? "bad" : "warn";
  return (
    <div className="mb-4">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <h1 className="num text-xl font-semibold text-ink">{sgt(run.evaluated_at)}</h1>
        <Tip content={{ title: run.status.label, what: run.status.description }}>
          <span tabIndex={0} className="focus-ring rounded-full">
            <Badge tone={tone}>{run.status.label}{run.status.detail && ` · ${run.status.detail}`}</Badge>
          </span>
        </Tip>
        {!run.is_complete && <Badge tone="neutral" icon={false}>Partial / debug run</Badge>}
        {run.graph_worktree_dirty && (
          <Badge tone="warn" title="The recorded commit does not fully identify the code that ran.">
            Uncommitted code changes were present when this evaluation ran
          </Badge>
        )}
      </div>
      <p className="mt-1 text-xs text-ink-3">
        <span className="font-mono">{run.run_id}</span> · commit <span className="font-mono">{shortHash(run.graph_commit)}</span>
        {run.chat_model_id && <> · {run.chat_model_id}</>} · {run.source} ·{" "}
        {plural(data.summary.overall.counts?.total_questions ?? data.rows.length, "check")}
      </p>
      <Disclosure title="Run details" className="mt-2"><RunDetails run={run} /></Disclosure>
    </div>
  );
}

export default function RunPage() {
  const source = useSource();
  const { datasetId = "", runId = "" } = useParams();
  const [params, setParams] = useQueryState();
  const query = useRun(source, runId);
  const tab = TABS.find((item) => item.id === params.get("tab"))?.id ?? "overview";
  const crumbs = [
    { label: "Datasets", to: paths.datasets(source) },
    { label: query.data?.dataset.label ?? "Dataset", to: paths.dataset(source, datasetId) },
    { label: query.data ? sgt(query.data.run.evaluated_at, { seconds: false }) : "Run" },
  ];
  if (query.isPending) return <Page crumbs={crumbs}><Loading label="Loading run" /></Page>;
  if (query.isError) return <Page crumbs={crumbs}><ErrorPanel error={query.error} /></Page>;
  const data = query.data;
  const issues = data.problems.reduce((total, problem) => total + (problem.target === "diagnostics" ? problem.count : 0), 0);

  return (
    <Page crumbs={crumbs}>
      <RunProvider data={data}>
        <RunHeader data={data} />
        <div role="tablist" aria-label="Run views" className="mb-4 flex gap-1 border-b border-line">
          {TABS.map((item) => (
            <button key={item.id} role="tab" type="button" aria-selected={tab === item.id}
              onClick={() => setParams({ tab: item.id === "overview" ? null : item.id, panel: null, focus: null, scope: null })}
              className={cx("focus-ring -mb-px border-b-2 px-3 py-2 text-[13px] font-medium",
                tab === item.id ? "border-accent text-ink" : "border-transparent text-ink-2 hover:text-ink")}>
              {item.label}
              {item.id === "diagnostics" && issues > 0 && (
                <span className="num ml-1.5 inline-flex items-center gap-0.5 rounded-full bg-tint-warn px-1.5 text-[11px] text-ink-warn">
                  <AlertTriangle size={10} /> {issues}
                </span>
              )}
            </button>
          ))}
        </div>
        {tab === "overview" && <Overview />}
        {tab === "analysis" && <Analysis />}
        {tab === "diagnostics" && <Diagnostics />}
        <PanelHost />
      </RunProvider>
    </Page>
  );
}
