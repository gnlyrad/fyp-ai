import { ArrowLeft, ChevronLeft, ChevronRight } from "lucide-react";
import { Link, useLocation, useParams } from "react-router-dom";
import { Page } from "../components/Page";
import { Badge, Card, Disclosure, ErrorPanel, FieldName, Json, Loading, MetricTip, SectionTitle, cx } from "../components/ui";
import { useResult, useRun, type ResultDetail, type Turn } from "../lib/api";
import { pct, sgt } from "../lib/format";
import { useMetaApi } from "../lib/meta";
import { paths, useSource, type ResultNavState } from "../lib/url";

type Mark = "expected" | "selected" | "both";

const MARK: Record<Mark, { label: string; border: string; dot: string }> = {
  expected: { label: "Expected evidence", border: "border-o-unclear", dot: "bg-o-unclear" },
  selected: { label: "Model-selected evidence", border: "border-o-correct", dot: "bg-o-correct" },
  both: { label: "Expected and selected", border: "border-o-both", dot: "bg-o-both" },
};

function markOf(turn: Turn): Mark | null {
  return turn.expected && turn.selected ? "both" : turn.expected ? "expected" : turn.selected ? "selected" : null;
}

function Legend() {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2">
      {(Object.keys(MARK) as Mark[]).map((mark) => (
        <span key={mark} className="inline-flex items-center gap-1.5">
          <span className={cx("h-3 w-1 rounded-full", MARK[mark].dot)} /> {MARK[mark].label}
        </span>
      ))}
    </div>
  );
}

function Excerpts({ turns, empty }: { turns: Turn[]; empty: string }) {
  if (!turns.length) return <p className="text-sm italic text-ink-3">{empty}</p>;
  return (
    <ul className="space-y-2">
      {turns.map((turn) => (
        <li key={turn.id}>
          <a href={`#turn-${turn.id}`} className="focus-ring block rounded-md border border-line bg-surface-2 px-3 py-2 hover:border-accent/60">
            <div className="text-[11px] font-medium text-ink-3">{turn.speaker} · turn {turn.id}</div>
            <div className="text-[13px] text-ink">“{turn.text}”</div>
          </a>
        </li>
      ))}
    </ul>
  );
}

function Verdicts({ result }: { result: ResultDetail }) {
  const { status, presentation } = useMetaApi();
  const { row } = result;
  const failedRun = row.execution_status === "prediction_failure";
  const expected = row.family === "compliance"
    ? <span className="text-2xl font-semibold text-ink">{row.gold_verdict}</span>
    : row.expected_value === null
      ? <span className="text-lg font-semibold text-ink-2">{status("extraction_gold_value_status", "NOT_STATED")}</span>
      : <span className="text-lg font-semibold text-ink">“{row.expected_value}”</span>;
  let predicted;
  if (failedRun) {
    predicted = <span className="text-lg font-semibold text-ink">No usable result</span>;
  } else if (row.family === "compliance") {
    predicted = <span className="text-2xl font-semibold text-ink">{row.predicted_verdict}</span>;
  } else {
    predicted = <span className="text-lg font-semibold text-ink">{row.outcome.label}</span>;
  }
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <Card className="p-4">
        <SectionTitle>Expected (ground truth)</SectionTitle>
        {expected}
        {row.family === "extraction" && row.value_type && <div className="mt-1 text-xs text-ink-3">Value type: {row.value_type}</div>}
        {row.family === "extraction" && row.gold_value_status === "NOT_STATED" && (
          <p className="mt-1 text-xs text-ink-3">{presentation.statuses.extraction_gold_value_status.NOT_STATED.description}</p>
        )}
      </Card>
      <Card className="p-4">
        <SectionTitle aside={<Badge tone={row.outcome.tone}>{row.outcome.label}</Badge>}>Pipeline result</SectionTitle>
        {predicted}
        {failedRun && (
          <div className="mt-1 text-xs text-ink-2">
            {status("failure_category", row.failure_category)}
            {result.failure_message && <span className="text-ink-3"> · {result.failure_message}</span>}
          </div>
        )}
        {row.family === "compliance" && row.compliance_outcome_status === "abstention" && (
          <p className="mt-1 text-xs text-ink-3">{presentation.statuses.compliance_outcome_status.abstention.description}</p>
        )}
      </Card>
    </div>
  );
}

function EvidenceMatch({ result }: { result: ResultDetail }) {
  const { row } = result;
  const family = row.family;
  const stat = (label: string, metricId: string, value: string) => (
    <div className="rounded-md border border-line px-3 py-2">
      <div className="flex items-center gap-1 text-xs text-ink-2">{label}<MetricTip id={metricId} /></div>
      <div className="num text-lg font-semibold text-ink">{value}</div>
    </div>
  );
  const asPct = (value: number | null) => (value === null ? "—" : pct(value * 100));
  return (
    <Card className="p-4">
      <SectionTitle>Evidence match</SectionTitle>
      <div className="grid grid-cols-3 gap-2">
        {stat("Selected evidence relevance", `${family}.evidence_precision`, asPct(row.evidence_precision))}
        {stat("Expected evidence recovered", `${family}.evidence_recall`, asPct(row.evidence_recall))}
        {stat("Exact evidence match", `${family}.evidence_exact_set_match`,
          row.evidence_exact_match === null ? "—" : row.evidence_exact_match === 1 ? "Yes" : "No")}
      </div>
      <p className="mt-3 text-[13px] text-ink">{result.explanation}</p>
    </Card>
  );
}

function Transcript({ turns }: { turns: Turn[] }) {
  return (
    <Card className="p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-ink">Transcript</h2>
        <Legend />
      </div>
      <ol className="space-y-1">
        {turns.map((turn) => {
          const mark = markOf(turn);
          return (
            <li key={turn.id} id={`turn-${turn.id}`}
              className={cx("scroll-mt-20 rounded-r-md border-l-[3px] py-1.5 pl-3 pr-2 target:bg-surface-2",
                mark ? MARK[mark].border : "border-transparent")}>
              <div className="flex flex-wrap items-center gap-2 text-[11px]">
                <span className="font-medium text-ink-2">Turn {turn.id} · {turn.speaker}</span>
                {mark && <span className="font-medium text-ink-3">{MARK[mark].label}</span>}
              </div>
              <div className={cx("text-[13px]", mark ? "text-ink" : "text-ink-2")}>{turn.text}</div>
            </li>
          );
        })}
      </ol>
    </Card>
  );
}

export default function ResultPage() {
  const source = useSource();
  const { datasetId = "", runId = "", callId = "", questionId = "" } = useParams();
  const location = useLocation();
  const nav = location.state as ResultNavState | null;
  const run = useRun(source, runId);
  const result = useResult(source, runId, callId, questionId);
  const { status } = useMetaApi();

  const runPath = paths.run(source, datasetId, runId);
  const back = nav?.from ?? `${runPath}?tab=analysis`;
  const list: [string, string][] = nav?.list ?? run.data?.rows.map((row) => [row.call_id, row.question_id]) ?? [];
  const position = list.findIndex(([c, q]) => c === callId && q === questionId);
  const neighbour = (offset: number) => {
    const target = list[position + offset];
    return target && position >= 0 ? paths.result(source, datasetId, runId, target[0], target[1]) : null;
  };
  const crumbs = [
    { label: "Datasets", to: paths.datasets(source) },
    { label: run.data?.dataset.label ?? "Dataset", to: paths.dataset(source, datasetId) },
    { label: run.data ? sgt(run.data.run.evaluated_at, { seconds: false }) : "Run", to: runPath },
    { label: `${callId} · ${questionId}` },
  ];
  const prev = neighbour(-1);
  const next = neighbour(1);

  return (
    <Page crumbs={crumbs}>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-[13px]">
        <Link to={back} className="focus-ring inline-flex items-center gap-1 rounded font-medium text-accent-ink hover:underline">
          <ArrowLeft size={14} /> Back{nav?.listLabel ? ` to ${nav.listLabel}` : " to analysis"}
        </Link>
        {position >= 0 && list.length > 1 && (
          <div className="flex items-center gap-2 text-ink-2">
            <span className="num text-xs">{position + 1} of {list.length}{nav?.listLabel ? ` · ${nav.listLabel}` : ""}</span>
            {[{ to: prev, icon: ChevronLeft, label: "Previous" }, { to: next, icon: ChevronRight, label: "Next" }].map(({ to, icon: Icon, label }) =>
              to ? (
                <Link key={label} to={to} state={nav} replace aria-label={label}
                  className="focus-ring rounded-md border border-line p-1 hover:bg-surface-2"><Icon size={14} /></Link>
              ) : (
                <span key={label} className="rounded-md border border-line p-1 opacity-40"><Icon size={14} /></span>
              ),
            )}
          </div>
        )}
      </div>

      {result.isPending && <Loading label="Loading result" />}
      {result.isError && <ErrorPanel error={result.error} />}
      {result.data && (() => {
        const data = result.data;
        const expectedTurns = data.turns.filter((turn) => turn.expected);
        const selectedTurns = data.turns.filter((turn) => turn.selected);
        return (
          <div className="space-y-4">
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-xl font-semibold text-ink">{callId} · {questionId}</h1>
                <Badge tone="neutral" icon={false}>{data.row.family === "compliance" ? "Compliance" : "Structured extraction"}</Badge>
                {data.row.requirement_type && <Badge tone="neutral" icon={false}>{data.row.requirement_type.toLowerCase()}</Badge>}
                {data.row.field && <Badge tone="neutral" icon={false}><FieldName field={data.row.field} /></Badge>}
              </div>
              <p className="mt-1 text-base text-ink">{data.question.question}</p>
            </div>

            <Verdicts result={data} />

            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <Card className="p-4">
                <SectionTitle>Expected evidence</SectionTitle>
                <Excerpts turns={expectedTurns} empty="No evidence is expected for this check." />
              </Card>
              <Card className="p-4">
                <SectionTitle>Model-selected evidence</SectionTitle>
                <Excerpts turns={selectedTurns}
                  empty={data.row.execution_status === "prediction_failure" ? "No evidence: the prediction failed." : "The model selected no evidence."} />
                {data.selected_quotes.length > 0 && (
                  data.row.family === "extraction" ? (
                    <div className="mt-3">
                      <div className="mb-1 text-xs font-medium text-ink-2">Quoted text the evaluator searched for the value</div>
                      <ul className="space-y-1">
                        {data.selected_quotes.map((quote, index) => (
                          <li key={index} className="rounded border border-line px-2 py-1 font-mono text-xs text-ink">{quote}</li>
                        ))}
                      </ul>
                    </div>
                  ) : (
                    <Disclosure title="Exact quoted spans" className="mt-3">
                      <ul className="space-y-1">
                        {data.selected_quotes.map((quote, index) => (
                          <li key={index} className="rounded border border-line px-2 py-1 font-mono text-xs text-ink-2">{quote}</li>
                        ))}
                      </ul>
                    </Disclosure>
                  )
                )}
              </Card>
            </div>

            <EvidenceMatch result={data} />

            {data.reasoning && (
              <Card className="p-4">
                <SectionTitle>Model reasoning</SectionTitle>
                <p className="text-[13px] text-ink">{data.reasoning}</p>
              </Card>
            )}

            <Transcript turns={data.turns} />

            <Card className="p-4">
              <Disclosure title="Technical details">
                <dl className="mb-3 grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
                  {[
                    ["Field (internal name)", data.row.field ?? "—"],
                    ["Execution status", `${status("execution_status", data.row.execution_status)} (${data.row.execution_status})`],
                    ["Failure category", data.row.failure_category ?? "—"],
                    ["Raw pipeline verdict", data.row.native_verdict ?? "—"],
                    ["Compliance outcome", data.row.compliance_outcome_status ?? "—"],
                    ["Decision category", data.row.confusion_category ?? "—"],
                    ["Breach outcome", data.row.breach_outcome ?? "—"],
                    ["Extraction status", data.row.extraction_evidence_status ?? data.row.gold_value_status ?? "—"],
                    ["Evaluator support", data.row.evaluator_support ?? "—"],
                    ["Evidence status", data.row.evidence_status ?? "—"],
                    ["Expected / selected turns", `${JSON.stringify(data.row.gold_turns)} / ${JSON.stringify(data.row.predicted_turns)}`],
                  ].map(([label, value]) => (
                    <div key={label} className="flex gap-2">
                      <dt className="w-44 shrink-0 text-ink-3">{label}</dt>
                      <dd className="font-mono text-ink-2">{value}</dd>
                    </div>
                  ))}
                </dl>
                <div className="space-y-3">
                  <Disclosure title="Score record"><Json value={data.score} /></Disclosure>
                  <Disclosure title="Normalized prediction"><Json value={data.prediction} /></Disclosure>
                  <Disclosure title="Ground truth"><Json value={data.ground_truth} /></Disclosure>
                </div>
              </Disclosure>
            </Card>
          </div>
        );
      })()}
    </Page>
  );
}
