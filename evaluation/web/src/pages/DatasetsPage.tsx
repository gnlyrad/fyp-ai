import { ArrowRight, Database } from "lucide-react";
import { Link } from "react-router-dom";
import { Page } from "../components/Page";
import { Card, Empty, ErrorPanel, Loading } from "../components/ui";
import { useDatasets } from "../lib/api";
import { plural, sgt, shortHash } from "../lib/format";
import { paths, useSource } from "../lib/url";

export default function DatasetsPage() {
  const source = useSource();
  const datasets = useDatasets(source);
  return (
    <Page>
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <h1 className="text-xl font-semibold text-ink">Datasets</h1>
        <p className="text-xs text-ink-3">
          {source === "shared" ? "Published benchmark snapshots (Supabase)" : "Datasets and runs on this machine (evaluation/)"}
        </p>
      </div>
      {datasets.isPending && <Loading />}
      {datasets.isError && (
        <ErrorPanel error={datasets.error}
          action={source === "shared" && (
            <Link to={paths.datasets("local")} className="text-sm font-medium text-accent-ink hover:underline">Switch to Local</Link>
          )} />
      )}
      {datasets.data && !datasets.data.length && (
        <Card><Empty>No datasets are available in this source.</Empty></Card>
      )}
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {datasets.data?.map((dataset) => (
          <Link key={dataset.dataset_id} to={paths.dataset(source, dataset.dataset_id)}
            className="focus-ring group rounded-lg border border-line bg-surface p-4 transition hover:border-accent/60 hover:shadow-sm">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-2">
                <Database size={16} className="text-accent" />
                <span className="text-base font-semibold text-ink">{dataset.label}</span>
              </div>
              <ArrowRight size={16} className="text-ink-3 transition group-hover:translate-x-0.5 group-hover:text-ink" />
            </div>
            <div className="mt-2 text-[13px] text-ink-2">
              {plural(dataset.call_count, "call")} · {plural(dataset.question_count, "question")}
              <span className="text-ink-3"> ({dataset.compliance_questions} compliance, {dataset.extraction_questions} extraction)</span>
            </div>
            <div className="mt-1 text-[13px] text-ink-2">
              {plural(dataset.run_count ?? 0, "run")}
              {dataset.latest_run_at && <span className="text-ink-3"> · latest {sgt(dataset.latest_run_at, { seconds: false })}</span>}
            </div>
            <div className="mt-3 font-mono text-[11px] text-ink-3">snapshot {shortHash(dataset.dataset_sha256, 12)}</div>
          </Link>
        ))}
      </div>
    </Page>
  );
}
