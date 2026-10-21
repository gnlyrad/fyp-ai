import { ArrowRight, AudioLines } from "lucide-react";
import { Link } from "react-router-dom";
import { Page } from "../../components/Page";
import { Card, Empty, ErrorPanel, Loading } from "../../components/ui";
import { plural, sgt } from "../../lib/format";
import { sttPaths, useSttDatasets } from "../../lib/stt";

export default function SttDatasetsPage() {
  const datasets = useSttDatasets();
  return (
    <Page section="stt">
      <div className="mb-4 flex items-baseline justify-between gap-3">
        <h1 className="text-xl font-semibold text-ink">Speech-to-text datasets</h1>
        <p className="text-xs text-ink-3">Transcribe results exported from fyp-be (evaluation/stt_runs)</p>
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
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {datasets.data?.map((dataset) => (
          <Link key={dataset.dataset_id} to={sttPaths.dataset(dataset.dataset_id)}
            className="focus-ring group rounded-lg border border-line bg-surface p-4 transition hover:border-accent/60 hover:shadow-sm">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-center gap-2">
                <AudioLines size={16} className="text-accent" />
                <span className="text-base font-semibold text-ink">{dataset.label}</span>
              </div>
              <ArrowRight size={16} className="text-ink-3 transition group-hover:translate-x-0.5 group-hover:text-ink" />
            </div>
            <div className="mt-2 text-[13px] text-ink-2">
              {plural(dataset.run_count, "run")}
              {dataset.latest_run_at && <span className="text-ink-3"> · latest {sgt(dataset.latest_run_at, { seconds: false })}</span>}
            </div>
            <div className="mt-1 text-[13px] text-ink-3">{plural(dataset.dataset_call_count, "call")} in the dataset</div>
          </Link>
        ))}
      </div>
    </Page>
  );
}
