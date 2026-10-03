import { CheckCircle2 } from "lucide-react";
import { useParams } from "react-router-dom";
import { DataTable, type Column } from "../../components/DataTable";
import { Card, Empty, ErrorPanel, Help, Loading, MetricLabel, MetricTip, SectionTitle, cx } from "../../components/ui";
import { useDiagnostics, type DiagnosticError } from "../../lib/api";
import { fraction, ratioPct } from "../../lib/format";
import { useMetaApi } from "../../lib/meta";
import { useQueryState, useSource } from "../../lib/url";
import { useRunView } from "./context";

const EVIDENCE_ROWS: [string, string, string][] = [
  ["Evidence precision", "compliance.evidence_precision", "extraction.evidence_precision"],
  ["Evidence recall", "compliance.evidence_recall", "extraction.evidence_recall"],
  ["Exact evidence match", "compliance.evidence_exact_set_match", "extraction.evidence_exact_set_match"],
  ["Unmatched evidence references", "compliance.invalid_evidence_reference_rate", "extraction.invalid_evidence_reference_rate"],
  ["Raw UNCLEAR outputs", "compliance.native_unclear_outputs", "extraction.native_unclear_outputs"],
];

function EvidenceMetrics() {
  const { data } = useRunView();
  const { metric } = useMetaApi();
  const node = data.summary.overall;
  const cell = (id: string) => (
    <td className="px-3 py-1.5 text-right">
      <span className="inline-flex items-center gap-1.5">
        <span className="num font-medium text-ink">{ratioPct(metric(node, id))}</span>
        <span className="num text-xs text-ink-3">{fraction(metric(node, id))}</span>
        <MetricTip id={id} />
      </span>
    </td>
  );
  return (
    <Card className="p-4">
      <SectionTitle>Evidence selection (developer metrics)</SectionTitle>
      <table className="w-full text-[13px]">
        <thead>
          <tr className="border-b border-line text-xs text-ink-3">
            <th className="px-3 py-1.5 text-left font-medium">Metric</th>
            <th className="px-3 py-1.5 text-right font-medium">Compliance</th>
            <th className="px-3 py-1.5 text-right font-medium">Extraction</th>
          </tr>
        </thead>
        <tbody>
          {EVIDENCE_ROWS.map(([label, complianceId, extractionId]) => (
            <tr key={label} className="border-b border-line last:border-0">
              <td className="px-3 py-1.5 text-ink-2">{label}</td>
              {cell(complianceId)}
              {cell(extractionId)}
            </tr>
          ))}
          <tr>
            <td className="px-3 py-1.5 text-ink-2"><MetricLabel id="diagnostic.evidence_all_questions_precision" /></td>
            <td colSpan={2} className="px-3 py-1.5 text-right">
              <span className="num font-medium text-ink">{ratioPct(metric(node, "diagnostic.evidence_all_questions_precision"))}</span>
            </td>
          </tr>
        </tbody>
      </table>
      <Help>Evidence scores are averaged per check. When no evidence is expected, selecting none scores 100%.</Help>
    </Card>
  );
}

export default function Diagnostics() {
  const source = useSource();
  const { runId = "" } = useParams();
  const { open } = useRunView();
  const [params, setParams] = useQueryState();
  const diagnostics = useDiagnostics(source, runId);
  const category = params.get("category");

  const columns: Column<DiagnosticError>[] = [
    { key: "type", header: "Type", sort: (e) => e.label, cell: (e) => <span className="whitespace-nowrap text-ink">{e.label}</span> },
    { key: "call", header: "Call", sort: (e) => e.call_id ?? "", cell: (e) => <span className="whitespace-nowrap">{e.call_id ?? "—"}</span> },
    {
      key: "question", header: "Question", sort: (e) => e.question_id ?? "",
      cell: (e) => <span className={e.question_id ? "" : "text-ink-3"}>{e.question_id ?? "Call-level"}</span>,
    },
    {
      key: "message", header: "Message", className: "w-[60%]",
      cell: (e) => <span className="line-clamp-2 font-mono text-xs text-ink-2" title={e.message ?? ""}>{e.message}</span>,
    },
  ];

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
      <div className="space-y-3">
        {diagnostics.isPending && <Loading />}
        {diagnostics.isError && <ErrorPanel error={diagnostics.error} />}
        {diagnostics.data && (
          <>
            <div className="flex flex-wrap gap-2">
              <button type="button" onClick={() => setParams({ category: null })}
                className={cx("focus-ring rounded-full border px-3 py-1 text-xs font-medium",
                  !category ? "border-accent bg-tint-correct text-accent-ink" : "border-line text-ink-2 hover:text-ink")}>
                All <span className="num">{diagnostics.data.errors.length}</span>
              </button>
              {Object.entries(diagnostics.data.counts).map(([key, value]) => (
                <button key={key} type="button" onClick={() => setParams({ category: category === key ? null : key })}
                  className={cx("focus-ring rounded-full border px-3 py-1 text-xs font-medium",
                    category === key ? "border-accent bg-tint-correct text-accent-ink" : "border-line text-ink-2 hover:text-ink")}>
                  {diagnostics.data.errors.find((e) => e.category === key)?.label ?? key} <span className="num">{value}</span>
                </button>
              ))}
            </div>
            <Card>
              <DataTable id="errors" columns={columns}
                rows={diagnostics.data.errors.filter((e) => !category || e.category === category)}
                rowKey={(e) => String(e.index)} rowTitle="Show details"
                onRowClick={(e) => open(`error:${e.index}`)}
                empty={
                  <Empty>
                    <span className="inline-flex items-center gap-2"><CheckCircle2 size={16} className="text-o-correct" />
                      No operational errors or pipeline warnings were recorded.</span>
                  </Empty>
                } />
            </Card>
            <Help>
              Pipeline warnings are problems the pipeline recovered from; they are not model decisions. Question-level prediction
              failures are also counted in the Overview and Analysis tabs.
            </Help>
          </>
        )}
      </div>
      <EvidenceMetrics />
    </div>
  );
}
