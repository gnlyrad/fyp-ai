import { ArrowRight, CheckCircle2 } from "lucide-react";
import { ConfusionMatrix } from "../../components/ConfusionMatrix";
import { DataTable, type Column } from "../../components/DataTable";
import { OutcomeBar } from "../../components/OutcomeBar";
import { Card, FieldName, Kpi, MetricLabel, SectionTitle, Tip, cx } from "../../components/ui";
import type { ChecklistItem, Problem } from "../../lib/api";
import { count, fraction, ratioPct } from "../../lib/format";
import { useMetaApi } from "../../lib/meta";
import { useQueryState } from "../../lib/url";
import { useRunView, useSegments } from "./context";

function FamilyHeader({ family }: { family: "compliance" | "extraction" }) {
  const { presentation } = useMetaApi();
  const entry = presentation.families[family];
  return (
    <div className="mb-3 flex items-center gap-1.5">
      <h2 className="text-sm font-semibold text-ink">{entry.label}</h2>
      <Tip content={{ title: entry.label, what: entry.description }} />
    </div>
  );
}

function ComplianceSummary() {
  const { data, open } = useRunView();
  const { metric, drillCount } = useMetaApi();
  const segments = useSegments();
  const node = data.summary.overall;
  const total = drillCount(node, "compliance.all");
  const breaches = drillCount(node, "breach.all");
  const kpi = (id: string) => <Kpi id={id} ratio={metric(node, id)} onClick={() => open(`metric:${id}`)} />;
  return (
    <Card className="p-4">
      <FamilyHeader family="compliance" />
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        {kpi("compliance.decision_coverage")}
        {kpi("compliance.breach_recall")}
        {kpi("compliance.breach_precision")}
        {kpi("compliance.prediction_failure_rate")}
      </div>

      <div className="mt-4">
        <SectionTitle aside={<span className="inline-flex items-center gap-1.5"><MetricLabel id="compliance.abstention_rate" /><span className="num font-semibold text-ink">{ratioPct(metric(node, "compliance.abstention_rate"))}</span></span>}>Outcome of all {total} compliance checks</SectionTitle>
        <OutcomeBar label="Compliance outcomes" total={total} segments={segments.decisions(node, null)} />
      </div>

      <div className="mt-5 grid gap-5 md:grid-cols-2">
        <div>
          <SectionTitle aside={<MetricLabel id="compliance.confusion_matrix" />}>PASS/FAIL decisions vs ground truth</SectionTitle>
          <ConfusionMatrix node={node} onCell={(category) => open(`drill:confusion.${category}`)} />
          <button type="button" onClick={() => open("metric:compliance.binary_accuracy")}
            className="drill mt-2 inline-flex items-baseline gap-1.5 text-xs text-ink-2">
            <MetricLabel id="compliance.binary_accuracy" />
            <span className="num font-semibold text-ink">{ratioPct(metric(node, "compliance.binary_accuracy"))}</span>
            <span className="num text-ink-3">{fraction(metric(node, "compliance.binary_accuracy"))}</span>
          </button>
        </div>
        <div>
          <SectionTitle>What happened to the {breaches} actual breaches</SectionTitle>
          {breaches ? (
            <>
              <OutcomeBar label="Actual breach outcomes" total={breaches} segments={segments.breaches(node, null)} />
              <div className="mt-3 grid grid-cols-2 gap-2">
                <Kpi size="sm" id="compliance.automatic_breach_detection_rate"
                  ratio={metric(node, "compliance.automatic_breach_detection_rate")}
                  onClick={() => open("metric:compliance.automatic_breach_detection_rate")} />
                <Kpi size="sm" id="compliance.false_clearance_rate" ratio={metric(node, "compliance.false_clearance_rate")}
                  onClick={() => open("metric:compliance.false_clearance_rate")} />
              </div>
            </>
          ) : (
            <p className="text-sm text-ink-3">The ground truth contains no breaches in this run.</p>
          )}
        </div>
      </div>
    </Card>
  );
}

function ExtractionSummary() {
  const { data, open } = useRunView();
  const { metric, drillCount, drills } = useMetaApi();
  const segments = useSegments();
  const node = data.summary.overall;
  const expected = drillCount(node, "extraction.coverage_population");
  const fields = data.checklist.filter((item) => item.evaluation_family === "extraction");
  const byQuestion = data.summary.by_question_id;
  const countButton = (key: string, hideWhenZero = false) => {
    const value = drillCount(node, key);
    if (hideWhenZero && !value) return null;
    return (
      <li key={key} className="flex items-center justify-between py-1">
        <span className="text-ink-2">{drills[key].label}</span>
        <button type="button" disabled={!value} onClick={() => open(`drill:${key}`)}
          className={cx("num font-semibold text-ink", value ? "drill" : "cursor-default opacity-60")}>{value}</button>
      </li>
    );
  };
  const columns: Column<ChecklistItem>[] = [
    {
      key: "field", header: "Field", sort: (item) => item.id,
      cell: (item) => (
        <div className="flex min-w-0 items-baseline gap-1.5">
          <span className="font-medium text-ink">{item.id}</span>
          <FieldName field={item.field} className="text-ink-2" />
        </div>
      ),
    },
    {
      key: "coverage", header: <MetricLabel id="extraction.evidence_coverage" text="Coverage" />, align: "right",
      sort: (item) => metric(byQuestion[item.id], "extraction.evidence_coverage")?.percentage ?? null,
      cell: (item) => (
        <span className="whitespace-nowrap">
          <span className="font-medium text-ink">{ratioPct(metric(byQuestion[item.id], "extraction.evidence_coverage"))}</span>
          <span className="ml-1.5 text-xs text-ink-3">{fraction(metric(byQuestion[item.id], "extraction.evidence_coverage"))}</span>
        </span>
      ),
      onCell: (item) => open(`question:${item.id}`, { focus: "extraction" }),
    },
    {
      key: "failures", header: "Failures", align: "right",
      sort: (item) => count(byQuestion[item.id], "extraction.counts.prediction_failures"),
      cell: (item) => <span className="num">{count(byQuestion[item.id], "extraction.counts.prediction_failures")}</span>,
      onCell: (item) => open(`question:${item.id}`, { focus: "failures" }),
      cellEnabled: (item) => count(byQuestion[item.id], "extraction.counts.prediction_failures") > 0,
    },
  ];
  return (
    <Card className="p-4">
      <FamilyHeader family="extraction" />
      <div className="grid grid-cols-2 gap-2">
        <Kpi id="extraction.evidence_coverage" ratio={metric(node, "extraction.evidence_coverage")}
          caption={`${fraction(metric(node, "extraction.evidence_coverage")).replace(" / ", " of ")} expected values located`}
          onClick={() => open("metric:extraction.evidence_coverage")} />
        <Kpi id="extraction.prediction_failure_rate" ratio={metric(node, "extraction.prediction_failure_rate")}
          onClick={() => open("metric:extraction.prediction_failure_rate")} />
      </div>
      <div className="mt-4">
        <SectionTitle>Outcome of {expected} expected values</SectionTitle>
        <OutcomeBar label="Extraction outcomes" total={expected} segments={segments.extraction(node, null)} />
      </div>
      <ul className="mt-3 divide-y divide-line border-y border-line text-[13px]">
        {countButton("extraction.present")}
        {countButton("extraction.not_stated")}
        {countButton("extraction.prediction_failure")}
        {countButton("extraction.unsupported", true)}
      </ul>
      <div className="mt-4">
        <SectionTitle>By field</SectionTitle>
        <DataTable id="fields" columns={columns} rows={fields} rowKey={(item) => item.id}
          rowTitle="Open this field across all calls"
          onRowClick={(item) => open(`question:${item.id}`, { focus: "extraction" })} />
        {!fields.length && <p className="text-sm text-ink-3">No extraction questions in this dataset.</p>}
      </div>
    </Card>
  );
}

function WhereToLook() {
  const { data, open } = useRunView();
  const [, setParams] = useQueryState();
  const go = (problem: Problem) =>
    problem.target === "cases"
      ? open(`drill:${problem.drill}`)
      : setParams({ tab: "diagnostics", category: problem.category, panel: null, focus: null, scope: null });
  return (
    <Card className="p-4">
      <SectionTitle>Where to look</SectionTitle>
      {data.problems.length ? (
        <ul className="divide-y divide-line">
          {data.problems.map((problem) => (
            <li key={`${problem.family}-${problem.label}`}>
              <button type="button" onClick={() => go(problem)}
                className="focus-ring group flex w-full items-center gap-3 rounded px-1 py-1.5 text-left text-[13px] hover:bg-surface-2">
                <span className="w-20 shrink-0 text-xs capitalize text-ink-3">{problem.family}</span>
                <span className="flex-1 text-ink">{problem.label}</span>
                <span className="num font-semibold text-ink">{problem.count}</span>
                <ArrowRight size={14} className="text-ink-3 opacity-50 group-hover:opacity-100" />
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="flex items-center gap-2 text-sm text-ink-2">
          <CheckCircle2 size={16} className="text-o-correct" />
          Nothing to investigate: no missed breaches, false alarms, UNCLEAR results, failures or pipeline warnings.
        </p>
      )}
    </Card>
  );
}

export default function Overview() {
  const { data } = useRunView();
  const hasCompliance = Boolean(data.summary.overall.compliance);
  const hasExtraction = Boolean(data.summary.overall.extraction);
  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1.45fr)_minmax(0,1fr)]">
      <div className="space-y-4">
        {hasCompliance && <ComplianceSummary />}
      </div>
      <div className="space-y-4">
        <WhereToLook />
        {hasExtraction && <ExtractionSummary />}
      </div>
    </div>
  );
}
