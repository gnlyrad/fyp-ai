import { DataTable, type Column } from "../../components/DataTable";
import { Card, FieldName, Help, MetricLabel, cx } from "../../components/ui";
import type { ChecklistItem, SummaryNode } from "../../lib/api";
import { count, fraction, ratioPct } from "../../lib/format";
import { useMetaApi } from "../../lib/meta";
import { useQueryState } from "../../lib/url";
import { useRunView, type Focus } from "./context";

function Fraction({ ratio, suffix }: { ratio: ReturnType<ReturnType<typeof useMetaApi>["metric"]>; suffix?: string }) {
  if (!ratio || !ratio.denominator) return <span className="text-ink-3">—</span>;
  return (
    <span className="whitespace-nowrap">
      <span className="font-medium text-ink">{ratioPct(ratio)}</span>
      <span className="ml-1.5 text-xs text-ink-3">{fraction(ratio)}{suffix}</span>
    </span>
  );
}

function Count({ value }: { value: number }) {
  return <span className={cx("num", value ? "font-medium text-ink" : "text-ink-3")}>{value}</span>;
}

function ByCall() {
  const { data, callIds, open } = useRunView();
  const { metric, drillCount } = useMetaApi();
  const node = (callId: string): SummaryNode => data.summary.by_call_id[callId] ?? {};
  const focus = (key: Focus) => (callId: string) => open(`call:${callId}`, { focus: key });
  const failures = (callId: string) => count(node(callId), "counts.execution_status.prediction_failure");
  const columns: Column<string>[] = [
    { key: "call", header: "Call", sort: (id) => id, cell: (id) => <span className="font-medium text-ink">{id}</span> },
    {
      key: "decided", header: <MetricLabel id="compliance.decision_coverage" />, align: "right",
      sort: (id) => metric(node(id), "compliance.decision_coverage")?.percentage ?? null,
      cell: (id) => <Fraction ratio={metric(node(id), "compliance.decision_coverage")} />,
      onCell: focus("decision"), cellTitle: "Show decision outcomes for this call",
    },
    {
      key: "breaches", header: <MetricLabel id="compliance.automatic_breach_detection_rate" />, align: "right",
      sort: (id) => metric(node(id), "compliance.automatic_breach_detection_rate")?.percentage ?? null,
      cell: (id) => drillCount(node(id), "breach.all")
        ? <span className="whitespace-nowrap"><span className="font-medium text-ink">{drillCount(node(id), "breach.detected")} / {drillCount(node(id), "breach.all")}</span><span className="ml-1 text-xs text-ink-3">caught</span></span>
        : <span className="text-xs text-ink-3">No breaches</span>,
      onCell: focus("breaches"), cellEnabled: (id) => drillCount(node(id), "breach.all") > 0,
      cellTitle: "Show what happened to each actual breach",
    },
    {
      key: "unclear", header: "UNCLEAR", align: "right", sort: (id) => drillCount(node(id), "compliance.abstention"),
      cell: (id) => <Count value={drillCount(node(id), "compliance.abstention")} />,
      onCell: focus("abstentions"), cellEnabled: (id) => drillCount(node(id), "compliance.abstention") > 0,
    },
    {
      key: "failures", header: "Failures", align: "right", sort: failures,
      cell: (id) => <Count value={failures(id)} />,
      onCell: focus("failures"), cellEnabled: (id) => failures(id) > 0,
    },
    {
      key: "extraction", header: <MetricLabel id="extraction.evidence_coverage" />, align: "right",
      sort: (id) => metric(node(id), "extraction.evidence_coverage")?.percentage ?? null,
      cell: (id) => <Fraction ratio={metric(node(id), "extraction.evidence_coverage")} />,
      onCell: focus("extraction"), cellTitle: "Show extraction results for this call",
    },
  ];
  return (
    <Card>
      <DataTable id="calls" columns={columns} rows={callIds} rowKey={(id) => id}
        rowTitle="Open this call" onRowClick={(id) => open(`call:${id}`)} />
    </Card>
  );
}

function ByQuestion() {
  const { data, open } = useRunView();
  const { metric, drillCount } = useMetaApi();
  const node = (id: string): SummaryNode => data.summary.by_question_id[id] ?? {};
  const focus = (key: Focus) => (item: ChecklistItem) => open(`question:${item.id}`, { focus: key });
  const compliance = data.checklist.filter((item) => item.evaluation_family === "compliance" && data.summary.by_question_id[item.id]);
  const extraction = data.checklist.filter((item) => item.evaluation_family === "extraction" && data.summary.by_question_id[item.id]);
  const failures = (item: ChecklistItem) => count(node(item.id), "counts.execution_status.prediction_failure");
  const question: Column<ChecklistItem> = {
    key: "question", header: "Question", sort: (item) => item.id, className: "w-[55%]",
    cell: (item) => (
      <div className="flex min-w-0 items-baseline gap-2">
        <span className="shrink-0 font-medium text-ink">{item.id}</span>
        {item.field && <FieldName field={item.field} className="shrink-0 font-medium text-ink-2" />}
        <span className="truncate text-ink-2" title={item.question}>{item.question}</span>
      </div>
    ),
  };
  const failuresColumn: Column<ChecklistItem> = {
    key: "failures", header: "Failures", align: "right", sort: failures,
    cell: (item) => <Count value={failures(item)} />, onCell: focus("failures"), cellEnabled: (item) => failures(item) > 0,
  };
  const complianceColumns: Column<ChecklistItem>[] = [
    question,
    {
      key: "decided", header: <MetricLabel id="compliance.decision_coverage" />, align: "right",
      sort: (item) => metric(node(item.id), "compliance.decision_coverage")?.percentage ?? null,
      cell: (item) => <Fraction ratio={metric(node(item.id), "compliance.decision_coverage")} />,
      onCell: focus("decision"),
    },
    {
      key: "recall", header: <MetricLabel id="compliance.breach_recall" />, align: "right",
      sort: (item) => metric(node(item.id), "compliance.breach_recall")?.percentage ?? null,
      cell: (item) => <Fraction ratio={metric(node(item.id), "compliance.breach_recall")} />,
      onCell: focus("breaches"), cellEnabled: (item) => drillCount(node(item.id), "breach.all") > 0,
    },
    {
      key: "unclear", header: "UNCLEAR", align: "right", sort: (item) => drillCount(node(item.id), "compliance.abstention"),
      cell: (item) => <Count value={drillCount(node(item.id), "compliance.abstention")} />,
      onCell: focus("abstentions"), cellEnabled: (item) => drillCount(node(item.id), "compliance.abstention") > 0,
    },
    failuresColumn,
  ];
  const extractionColumns: Column<ChecklistItem>[] = [
    question,
    {
      key: "located", header: <MetricLabel id="extraction.evidence_coverage" />, align: "right",
      sort: (item) => metric(node(item.id), "extraction.evidence_coverage")?.percentage ?? null,
      cell: (item) => <Fraction ratio={metric(node(item.id), "extraction.evidence_coverage")} />,
      onCell: focus("extraction"),
    },
    {
      key: "notlocated", header: "Not located", align: "right", sort: (item) => drillCount(node(item.id), "extraction.not_found"),
      cell: (item) => <Count value={drillCount(node(item.id), "extraction.not_found")} />,
      onCell: focus("extraction"), cellEnabled: (item) => drillCount(node(item.id), "extraction.not_found") > 0,
    },
    failuresColumn,
    {
      key: "novalue", header: "No value expected", align: "right", sort: (item) => drillCount(node(item.id), "extraction.not_stated"),
      cell: (item) => <Count value={drillCount(node(item.id), "extraction.not_stated")} />,
    },
  ];
  return (
    <div className="space-y-4">
      {compliance.length > 0 && (
        <Card>
          <h3 className="border-b border-line px-3 py-2 text-sm font-semibold text-ink">Compliance questions</h3>
          <DataTable id="cq" columns={complianceColumns} rows={compliance} rowKey={(item) => item.id}
            rowTitle="Open this question across all calls" onRowClick={(item) => open(`question:${item.id}`)} />
        </Card>
      )}
      {extraction.length > 0 && (
        <Card>
          <h3 className="border-b border-line px-3 py-2 text-sm font-semibold text-ink">Structured extraction</h3>
          <DataTable id="eq" columns={extractionColumns} rows={extraction} rowKey={(item) => item.id}
            rowTitle="Open this field across all calls" onRowClick={(item) => open(`question:${item.id}`)} />
        </Card>
      )}
    </div>
  );
}

export default function Analysis() {
  const [params, setParams] = useQueryState();
  const view = params.get("view") === "question" ? "question" : "call";
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div role="tablist" aria-label="Analyse by" className="inline-flex rounded-md border border-line bg-surface p-0.5 text-[13px]">
          {(["call", "question"] as const).map((option) => (
            <button key={option} role="tab" type="button" aria-selected={view === option}
              onClick={() => setParams({ view: option === "call" ? null : option })}
              className={cx("focus-ring rounded px-3 py-1 font-medium",
                view === option ? "bg-surface-2 text-ink shadow-sm" : "text-ink-2 hover:text-ink")}>
              By {option}
            </button>
          ))}
        </div>
        <Help>Click a row to open the whole {view}; click a number to open just that part of it.</Help>
      </div>
      {view === "call" ? <ByCall /> : <ByQuestion />}
    </div>
  );
}
