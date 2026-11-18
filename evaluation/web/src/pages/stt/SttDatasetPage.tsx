import { useParams, useNavigate } from "react-router-dom";
import { DataTable, type Column } from "../../components/DataTable";
import { Page } from "../../components/Page";
import { Badge, Card, ErrorPanel, Loading } from "../../components/ui";
import { sgt } from "../../lib/format";
import { datasetLabel, sttPaths, useSttDataset, type SttRunSummary } from "../../lib/stt";
import { isAvailable, rate, sdi } from "./parts";

export default function SttDatasetPage() {
  const { datasetId = "" } = useParams();
  const navigate = useNavigate();
  const data = useSttDataset(datasetId);
  const columns: Column<SttRunSummary>[] = [
    {
      key: "when", header: "Run", sort: (run) => run.collected_at ?? "",
      cell: (run) => (
        <div>
          <div className="font-medium text-ink">{sgt(run.collected_at, { seconds: false })}</div>
          <div className="font-mono text-[11px] text-ink-3">{run.run_id}</div>
        </div>
      ),
    },
    {
      key: "coverage", header: "Calls evaluated", align: "right", sort: (run) => run.calls_scored,
      cell: (run) => (
        <span className="num whitespace-nowrap">
          {run.calls_scored} of {run.dataset_call_count || "?"}
          {run.dataset_call_count > run.calls_scored && <span className="ml-1.5"><Badge tone="neutral" icon={false}>Partial</Badge></span>}
        </span>
      ),
    },
    {
      key: "wer", header: "Normalised WER", align: "right", sort: (run) => run.normalised_wer.wer,
      cell: (run) => <span><span className="font-medium text-ink">{rate(run.normalised_wer.wer)}</span><span className="ml-1.5 text-xs text-ink-3">{sdi(run.normalised_wer)}</span></span>,
    },
    {
      key: "entity", header: "Entity WER", align: "right", sort: (run) => (isAvailable(run.entity_wer) ? run.entity_wer.wer : null),
      cell: (run) => (isAvailable(run.entity_wer) ? rate(run.entity_wer.wer) : <span className="text-ink-3">not available</span>),
    },
    { key: "speaker", header: "Speaker attribution", align: "right", sort: (run) => run.speaker_attribution.accuracy, cell: (run) => rate(run.speaker_attribution.accuracy) },
    { key: "conf", header: "Avg. confidence", align: "right", sort: (run) => run.average_confidence.mean, cell: (run) => <span className="num">{run.average_confidence.mean ?? "—"}</span> },
    {
      key: "setup", header: "Transcribe setup",
      cell: (run) => <span className="text-xs text-ink-2">{run.vocabulary_name ? `vocabulary ${run.vocabulary_name}` : "no custom vocabulary"}</span>,
    },
  ];
  return (
    <Page section="stt" crumbs={[{ label: "Datasets", to: sttPaths.home() }, { label: data.data ? datasetLabel(datasetId) : datasetId }]}>
      {data.isPending && <Loading />}
      {data.isError && <ErrorPanel error={data.error} />}
      {data.data && (
        <>
          <h1 className="mb-1 text-xl font-semibold text-ink">{datasetLabel(datasetId)}</h1>
          <p className="mb-4 text-[13px] text-ink-2">Each row is one set of Transcribe results, scored offline against the dataset's reference transcripts.</p>
          <Card>
            <DataTable id="sttruns" columns={columns} rows={data.data.runs} rowKey={(run) => run.run_id}
              rowTitle="Open this run" onRowClick={(run) => navigate(sttPaths.run(datasetId, run.run_id))} />
          </Card>
        </>
      )}
    </Page>
  );
}
