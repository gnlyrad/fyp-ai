import { AlertTriangle } from "lucide-react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { Page } from "../components/Page";
import { Card, Disclosure, ErrorPanel, Help, Loading, MetricLabel, OperationLabel } from "../components/ui";
import { useComparison, useDataset, type Comparison, type RunSummary } from "../lib/api";
import { duration, fraction, money, ratioPct, sgt, shortHash, signedChange, signedPp, tokens } from "../lib/format";
import { paths, useSource } from "../lib/url";
import { RunStatusBadge } from "./DatasetPage";

function RunColumnHead({ label, run, to }: { label: string; run: RunSummary; to: string }) {
  return (
    <th scope="col" className="px-3 py-2 text-right align-bottom font-normal">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">{label}</div>
      <Link to={to} className="focus-ring num whitespace-nowrap rounded font-medium text-ink hover:underline">
        {sgt(run.evaluated_at, { seconds: false, year: false })}
      </Link>
      <div className="font-mono text-[11px] text-ink-3">{shortHash(run.graph_commit)}</div>
    </th>
  );
}

function NotComparable({ data, labelFor }: { data: Comparison; labelFor: (runId: string) => string }) {
  const legacy = data.unverifiable_run_ids;
  return (
    <Card className="mb-4 border-o-wrong/40 p-4">
      <div className="flex items-start gap-2 text-sm">
        <AlertTriangle size={16} className="mt-0.5 shrink-0 text-ink-wrong" />
        <div className="min-w-0">
          <div className="font-medium text-ink">Accuracy differences can't be shown for these runs</div>
          {legacy.length > 0 ? (
            <>
              <p className="mt-1 text-ink-2">
                {legacy.map(labelFor).join(" and ")} {legacy.length === 1 ? "was" : "were"} saved before runs recorded which
                checklist and metric definitions scored them, so there is no proof both runs were scored the same way.
                Time, token and cost differences below are still valid.
              </p>
              <p className="mt-2 text-ink-2">To make an older run comparable, re-score it. This needs no model calls and adds a new run next to the original:</p>
              {legacy.map((runId) => (
                <pre key={runId} className="mt-1 overflow-x-auto rounded-md border border-line bg-surface-2 px-3 py-1.5 font-mono text-xs text-ink">
                  python -B evaluation/scripts/rescore_run.py --source-run {runId}
                </pre>
              ))}
            </>
          ) : (
            <p className="mt-1 text-ink-2">
              The two runs differ in the dataset, checklist or scoring version, so their accuracy numbers don't measure the same thing.
            </p>
          )}
          <Disclosure title="Technical reasons" className="mt-3">
            <ul className="list-disc pl-5 text-xs text-ink-2">
              {data.incompatibilities.map((reason) => <li key={reason}>{reason}</li>)}
            </ul>
          </Disclosure>
        </div>
      </div>
    </Card>
  );
}

export default function ComparePage() {
  const source = useSource();
  const { datasetId = "" } = useParams();
  const [params] = useSearchParams();
  const dataset = useDataset(source, datasetId);
  const comparison = useComparison(source, params.get("a") ?? "", params.get("b") ?? "");
  const crumbs = [
    { label: "Datasets", to: paths.datasets(source) },
    { label: dataset.data?.dataset.label ?? "Dataset", to: paths.dataset(source, datasetId) },
    { label: "Compare runs" },
  ];
  if (comparison.isPending) return <Page crumbs={crumbs}><Loading label="Comparing runs" /></Page>;
  if (comparison.isError) return <Page crumbs={crumbs}><ErrorPanel error={comparison.error} /></Page>;
  const data = comparison.data;
  const labelFor = (runId: string) => sgt((runId === data.earlier.run_id ? data.earlier : data.later).evaluated_at, { seconds: false });
  const currency = data.later.operations.cost?.currency ?? data.earlier.operations.cost?.currency ?? "USD";
  const operationValue = (unit: "ms" | "tokens" | "currency", value: number | null) => {
    if (value === null) return "—";
    return unit === "ms" ? duration(value) : unit === "tokens" ? tokens(value) : money(value, currency);
  };
  const priceNote = [data.earlier, data.later].some((run) => run.operations.cost && run.operations.cost.amount === null);

  return (
    <Page crumbs={crumbs}>
      <h1 className="mb-1 text-xl font-semibold text-ink">Compare runs</h1>
      <p className="mb-4 text-[13px] text-ink-2">
        Differences are always the later run minus the earlier run.
      </p>
      <div className="mb-4 grid gap-3 sm:grid-cols-2">
        {([["Earlier run", data.earlier], ["Later run", data.later]] as const).map(([label, run]) => (
          <Card key={run.run_id} className="p-3">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">{label}</div>
            <Link to={paths.run(source, datasetId, run.run_id)} className="focus-ring num rounded font-medium text-ink hover:underline">
              {sgt(run.evaluated_at)}
            </Link>
            <div className="mt-1 flex flex-wrap items-center gap-1.5 text-xs text-ink-3">
              <span className="font-mono">{shortHash(run.graph_commit)}</span>
              {run.chat_model_id && <span>· {run.chat_model_id}</span>}
              <RunStatusBadge run={run} />
            </div>
          </Card>
        ))}
      </div>

      {!data.comparable && <NotComparable data={data} labelFor={labelFor} />}

      <Card className="mb-4">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-line text-xs text-ink-3">
              <th scope="col" className="px-3 py-2 text-left font-medium">Accuracy</th>
              <RunColumnHead label="Earlier" run={data.earlier} to={paths.run(source, datasetId, data.earlier.run_id)} />
              <RunColumnHead label="Later" run={data.later} to={paths.run(source, datasetId, data.later.run_id)} />
              <th scope="col" className="px-3 py-2 text-right align-bottom font-medium">Change</th>
            </tr>
          </thead>
          <tbody>
            {data.metrics.map((row) => (
              <tr key={row.metric_id} className="border-b border-line last:border-0">
                <td className="px-3 py-1.5 text-ink-2"><MetricLabel id={row.metric_id} /></td>
                {[row.earlier, row.later].map((ratio, index) => (
                  <td key={index} className="px-3 py-1.5 text-right">
                    <span className="num font-medium text-ink">{ratioPct(ratio)}</span>
                    <span className="num ml-1.5 text-xs text-ink-3">{fraction(ratio)}</span>
                  </td>
                ))}
                <td className="num px-3 py-1.5 text-right font-medium text-ink">{data.comparable ? signedPp(row.delta_pp) : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>

      <Card>
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-line text-xs text-ink-3">
              <th scope="col" className="px-3 py-2 text-left font-medium">Time and cost</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Earlier</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Later</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Change</th>
            </tr>
          </thead>
          <tbody>
            {data.operations.map((row) => (
              <tr key={row.id} className="border-b border-line last:border-0">
                <td className="px-3 py-1.5 text-ink-2"><OperationLabel id={row.id} /></td>
                <td className="num px-3 py-1.5 text-right font-medium text-ink">{operationValue(row.unit, row.earlier)}</td>
                <td className="num px-3 py-1.5 text-right font-medium text-ink">{operationValue(row.unit, row.later)}</td>
                <td className="num px-3 py-1.5 text-right font-medium text-ink">{signedChange(row.delta, row.unit, currency)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <div className="mt-2 space-y-1">
        <Help>
          Accuracy changes are percentage points. No pass/fail thresholds are applied, and whether higher is better depends on the metric.
        </Help>
        <Help>
          "—" in the time and cost rows means that run did not record it. Runs before this was tracked have no token or cost data.
          {priceNote && " Cost is blank where no model price was configured when the run was made (see model_pricing.json)."}
        </Help>
      </div>
    </Page>
  );
}
