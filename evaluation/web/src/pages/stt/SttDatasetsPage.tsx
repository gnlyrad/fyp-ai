import { ArrowRight, AudioLines, ChevronDown } from "lucide-react";
import { Link } from "react-router-dom";
import { Page } from "../../components/Page";
import { Card, Empty, ErrorPanel, Loading } from "../../components/ui";
import { plural, sgt } from "../../lib/format";
import { datasetLabel, NOISY_DATASET_SUFFIX, sttPaths, useSttDatasets, type SttDataset } from "../../lib/stt";

const MULTILINGUAL = "advisory_execution_multilingual_v1";
const EARLIER = ["advisory_execution_pilot_v1", "legacy_pilot_v1"];
const baseId = (id: string) => id.endsWith(NOISY_DATASET_SUFFIX) ? id.slice(0, -NOISY_DATASET_SUFFIX.length) : id;
const isMultilingual = (id: string) => [MULTILINGUAL + "_heldout", MULTILINGUAL + "_development"].includes(baseId(id));

function DatasetCard({ dataset, to, title }: { dataset: SttDataset; to: string; title?: string }) {
  return (
    <Link to={to} className="focus-ring group block rounded-lg border border-line bg-surface p-4 transition hover:border-accent/60 hover:shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <span className="text-base font-semibold text-ink">{title ?? datasetLabel(dataset.dataset_id)}</span>
        <ArrowRight size={16} className="mt-1 shrink-0 text-ink-3 transition group-hover:translate-x-0.5 group-hover:text-ink" />
      </div>
      <div className="mt-2 text-[13px] text-ink-2">
        {plural(dataset.dataset_call_count, "call")} · {plural(dataset.run_count, "run")}
      </div>
      {dataset.latest_run_at && <div className="mt-1 text-xs text-ink-3">Latest {sgt(dataset.latest_run_at, { seconds: false })}</div>}
    </Link>
  );
}

export default function SttDatasetsPage() {
  const datasets = useSttDatasets();
  const available = datasets.data ?? [];
  const hasMultilingual = available.some((dataset) => isMultilingual(dataset.dataset_id));
  const earlier = available.filter((dataset) => EARLIER.includes(baseId(dataset.dataset_id)))
    .sort((a, b) => EARLIER.indexOf(baseId(a.dataset_id)) - EARLIER.indexOf(baseId(b.dataset_id)) || a.dataset_id.localeCompare(b.dataset_id));
  const other = available.filter((dataset) => !isMultilingual(dataset.dataset_id) && !EARLIER.includes(baseId(dataset.dataset_id)))
    .sort((a, b) => (b.latest_run_at ?? "").localeCompare(a.latest_run_at ?? ""));
  return (
    <Page section="stt">
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <h1 className="text-xl font-semibold text-ink">Speech-to-text datasets</h1>
        <p className="text-xs text-ink-3">
          Local speech-to-text evaluation runs
        </p>
      </div>
      {datasets.isPending && <Loading />}
      {datasets.isError && <ErrorPanel error={datasets.error} />}
      {datasets.data && !datasets.data.length && (
        <Card>
          <Empty>
            No speech-to-text runs yet. Score a run in fyp-be, then export it with{" "}
            <span className="font-mono text-xs">python -B scripts/export_to_fyp_ai.py --run &lt;run id&gt;</span>.
          </Empty>
        </Card>
      )}
      {hasMultilingual && (
        <section aria-labelledby="multilingual-title">
          <h2 id="multilingual-title" className="mb-4 flex items-center gap-2 text-base font-semibold text-ink">
            <AudioLines size={18} className="text-accent" /> Advisory Execution · Multilingual V1
          </h2>
          <div className="grid gap-3">
            {[true, false].map((noisy) => (
              <section key={String(noisy)} aria-label={noisy ? "Noisy and channel-degraded datasets" : "Clean datasets"}>
                <div className="grid gap-3 md:grid-cols-2">
                  {["development", "heldout"].map((split) => {
                    const id = `${MULTILINGUAL}_${split}${noisy ? NOISY_DATASET_SUFFIX : ""}`;
                    const dataset = available.find((item) => item.dataset_id === id);
                    const title = `${split === "heldout" ? "Held-out" : "Development"} (${noisy ? "noisy / channel-degraded" : "clean"})`;
                    return dataset ? (
                      <DatasetCard key={id} dataset={dataset} to={sttPaths.dataset(id)} title={title} />
                    ) : (
                      <div key={id} className="min-h-[104px] rounded-lg border border-dashed border-line p-4">
                        <div className="text-base font-semibold text-ink-3">{title}</div>
                        <div className="mt-2 text-[13px] text-ink-3">No runs available</div>
                      </div>
                    );
                  })}
                </div>
              </section>
            ))}
          </div>
        </section>
      )}
      {other.length > 0 && (
        <section className={hasMultilingual ? "mt-6" : ""} aria-label="Other datasets">
          {hasMultilingual && <h2 className="mb-3 text-sm font-semibold text-ink-2">Other datasets</h2>}
          <div className="grid gap-3 md:grid-cols-2">
            {other.map((dataset) => <DatasetCard key={dataset.dataset_id} dataset={dataset} to={sttPaths.dataset(dataset.dataset_id)} />)}
          </div>
        </section>
      )}
      {earlier.length > 0 && (
        <details className="group/earlier mt-6 border-t border-line pt-4">
          <summary className="focus-ring flex cursor-pointer list-none items-center gap-2 text-sm font-semibold text-ink-2 [&::-webkit-details-marker]:hidden">
            <ChevronDown size={16} className="-rotate-90 transition group-open/earlier:rotate-0" />
            Earlier datasets <span className="font-normal text-ink-3">({earlier.length})</span>
          </summary>
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            {earlier.map((dataset) => <DatasetCard key={dataset.dataset_id} dataset={dataset} to={sttPaths.dataset(dataset.dataset_id)} />)}
          </div>
        </details>
      )}
    </Page>
  );
}
