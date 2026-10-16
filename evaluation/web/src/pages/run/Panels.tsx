import { ExternalLink } from "lucide-react";
import type { ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { Drawer } from "../../components/Drawer";
import { OutcomeBar } from "../../components/OutcomeBar";
import { TierSection, TieredResults } from "../../components/TieredResults";
import { Json, Kpi, MetricTip, SectionTitle, cx } from "../../components/ui";
import { useDiagnostics, type Row } from "../../lib/api";
import { plural } from "../../lib/format";
import { useMetaApi } from "../../lib/meta";
import { EXCEPTION_TIERS, splitTiers, uniqueRows, type TierId } from "../../lib/tiers";
import { paths, useQueryState, useSource } from "../../lib/url";
import { METRIC_POPULATIONS, parseScope, useRunView, useSegments, type Scope } from "./context";
import { EVIDENCE_PARTS, EvidencePanel, type EvidenceFamily, type EvidencePart } from "./EvidencePanel";

/** A summary block (KPIs and an outcome bar); the focused one is briefly highlighted. */
function Summary({ focused, title, children }: { focused: boolean; title: ReactNode; children: ReactNode }) {
  return (
    <section className={cx("rounded-md", focused && "flash")}>
      <SectionTitle>{title}</SectionTitle>
      {children}
    </section>
  );
}

const byFamily = (rows: Row[], family: Row["family"]) => rows.filter((row) => row.family === family);

/** Which block a `focus=` value points at (the URL values predate the tier layout). */
const FOCUS_BLOCK: Record<string, string> = {
  decision: "decision",
  breaches: "breaches",
  abstentions: "unclear",
  failures: "failure",
  extraction: "extraction",
};

/** Whole call or question. Exceptions come first, then the summaries, then successful results
    (collapsed while there is anything above them). The focused dimension is moved to the very
    top, in the DOM too, so visual, keyboard and screen-reader order agree. */
function EntityPanel({ scope, focus }: { scope: Exclude<Scope, null>; focus: string | null }) {
  const { rowsFor, nodeFor, open } = useRunView();
  const { drillCount, metric } = useMetaApi();
  const segments = useSegments();
  const rows = rowsFor(scope);
  const node = nodeFor(scope);
  const compliance = byFamily(rows, "compliance");
  const extraction = byFamily(rows, "extraction");
  const tiers = splitTiers(rows);
  const hasExceptions = EXCEPTION_TIERS.some((tier) => tiers[tier].length > 0);
  const show = scope.kind === "call" ? "question" : "call";
  const focusId = focus ? FOCUS_BLOCK[focus] : undefined;
  const breaches = drillCount(node, "breach.all");
  const kpis = (ids: string[], columns: string) => (
    <div className={cx("mb-3 grid gap-2", columns)}>
      {ids.map((id) => <Kpi key={id} size="sm" id={id} ratio={metric(node, id)} onClick={() => open(`metric:${id}`, { scope })} />)}
    </div>
  );

  const blocks: Record<string, ReactNode> = {};
  for (const tier of [...EXCEPTION_TIERS, "success"] as TierId[]) {
    blocks[tier] = (
      <TierSection key={tier} tier={tier} rows={tiers[tier]} label={scope.id} show={show}
        flash={focusId === tier} collapsible={tier === "success" && hasExceptions} />
    );
  }
  if (compliance.length) {
    blocks.decision = (
      <Summary key="decision" focused={focusId === "decision"} title={`Decisions (${compliance.length} compliance checks)`}>
        {kpis(["compliance.decision_coverage", "compliance.abstention_rate", "compliance.prediction_failure_rate"], "grid-cols-3")}
        <OutcomeBar label="Compliance outcomes" total={compliance.length} segments={segments.decisions(node, scope)} />
      </Summary>
    );
    blocks.breaches = (
      <Summary key="breaches" focused={focusId === "breaches"} title={`Actual breaches (${breaches})`}>
        {breaches ? (
          <>
            {kpis(["compliance.breach_recall", "compliance.automatic_breach_detection_rate"], "grid-cols-2")}
            <OutcomeBar label="Actual breach outcomes" total={breaches} segments={segments.breaches(node, scope)} />
          </>
        ) : <p className="text-sm text-ink-3">The ground truth has no breaches here.</p>}
      </Summary>
    );
  }
  if (extraction.length) {
    blocks.extraction = (
      <Summary key="extraction" focused={focusId === "extraction"} title={`Structured extraction (${extraction.length} checks)`}>
        {kpis(["extraction.evidence_recall", "extraction.prediction_failure_rate"], "grid-cols-2")}
        <OutcomeBar label="Extraction outcomes" total={extraction.length} segments={segments.extraction(extraction, scope)} />
      </Summary>
    );
  }

  const defaultOrder = ["missed", "unclear", "failure", "other", "decision", "breaches", "extraction", "success"];
  const order = [...(focusId ? [focusId] : []), ...defaultOrder.filter((id) => id !== focusId)];
  return <div className="space-y-6">{order.map((id) => blocks[id] ?? null)}</div>;
}

function MetricPanel({ id, scope }: { id: string; scope: Scope }) {
  const { rowsFor, nodeFor, scopeLabel } = useRunView();
  const { wording, metric, drillRows } = useMetaApi();
  const words = wording(id);
  const node = nodeFor(scope);
  const rows = uniqueRows((METRIC_POPULATIONS[id] ?? []).flatMap((key) => drillRows(rowsFor(scope), key)));
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-[minmax(0,180px)_1fr] items-start gap-4">
        <Kpi id={id} ratio={metric(node, id)} />
        <div className="space-y-1.5 text-[13px] text-ink-2">
          <p>{words.what}</p>
          <p><span className="font-medium text-ink">How it's calculated: </span>{words.calculation}</p>
          {words.important && <p><span className="font-medium text-ink">Important: </span>{words.important}</p>}
        </div>
      </div>
      <p className="text-xs text-ink-3">{scopeLabel(scope)} · the {rows.length} checks behind this number, exceptions first:</p>
      <TieredResults rows={rows} label={`${words.label} · ${scopeLabel(scope)}`} empty="No checks behind this number." />
    </div>
  );
}

function DrillPanel({ drill, scope }: { drill: string; scope: Scope }) {
  const { rowsFor, scopeLabel } = useRunView();
  const { drillRows, drills } = useMetaApi();
  const rows = drillRows(rowsFor(scope), drill);
  return <TieredResults rows={rows} label={`${drills[drill].label} · ${scopeLabel(scope)}`} />;
}

function DiagnosticPanel({ index }: { index: number }) {
  const source = useSource();
  const { datasetId = "", runId = "" } = useParams();
  const { open } = useRunView();
  const diagnostics = useDiagnostics(source, runId);
  const error = diagnostics.data?.errors.find((item) => item.index === index);
  if (!error) return <p className="text-sm text-ink-3">{diagnostics.isPending ? "Loading…" : "Diagnostic not found."}</p>;
  return (
    <div className="space-y-4 text-[13px]">
      {error.description && <p className="text-ink-2">{error.description}</p>}
      <div>
        <SectionTitle>Message</SectionTitle>
        <p className="rounded-md border border-line bg-surface-2 p-3 font-mono text-xs text-ink">{error.message}</p>
      </div>
      <div className="flex flex-wrap gap-3">
        {error.linkable && error.call_id && error.question_id && (
          <Link to={paths.result(source, datasetId, runId, error.call_id, error.question_id)}
            className="inline-flex items-center gap-1 font-medium text-accent-ink hover:underline">
            Open {error.call_id} · {error.question_id} <ExternalLink size={13} />
          </Link>
        )}
        {error.call_id && (
          <button type="button" onClick={() => open(`call:${error.call_id}`)} className="font-medium text-accent-ink hover:underline">
            Open {error.call_id}
          </button>
        )}
      </div>
      <div>
        <SectionTitle>Technical details</SectionTitle>
        <Json value={error.payload} />
      </div>
    </div>
  );
}

/** Renders whichever drawer the URL asks for. */
export function PanelHost() {
  const [params] = useQueryState();
  const { close, data, rowsFor, nodeFor, questions } = useRunView();
  const { wording, drills, drillCount, drillRows, fieldLabel, presentation } = useMetaApi();
  const panel = params.get("panel");
  const focus = params.get("focus");
  const scope = parseScope(params.get("scope"));
  if (!panel) return <Drawer open={false} onClose={close} title="">{null}</Drawer>;
  const [kind, ...rest] = panel.split(":");
  const id = rest.join(":");

  let title: ReactNode = id;
  let subtitle: ReactNode = null;
  let body: ReactNode = null;
  if (kind === "call" || kind === "question") {
    const entity = { kind, id } as Exclude<Scope, null>;
    const rows = rowsFor(entity);
    const field = questions[id]?.field;
    title = kind === "call" ? id : field ? `${id} · ${fieldLabel(field)}` : id;
    subtitle = kind === "call"
      ? `${plural(byFamily(rows, "compliance").length, "compliance check")} · ${plural(byFamily(rows, "extraction").length, "extraction check")}`
      : questions[id]?.question;
    body = data.summary[kind === "call" ? "by_call_id" : "by_question_id"][id]
      ? <EntityPanel scope={entity} focus={focus} />
      : <p className="text-sm text-ink-3">Not part of this run.</p>;
  } else if (kind === "metric") {
    title = <span className="inline-flex items-center gap-1.5">{wording(id).label} <MetricTip id={id} /></span>;
    subtitle = scope ? `Scope: ${scope.kind === "call" ? scope.id : `question ${scope.id}`}` : "Whole run";
    body = <MetricPanel id={id} scope={scope} />;
  } else if (kind === "drill" && drills[id]) {
    title = drills[id].label;
    const total = drills[id].count_path ? drillCount(nodeFor(scope), id) : drillRows(rowsFor(scope), id).length;
    subtitle = `${plural(total, "check")} · ${scope ? (scope.kind === "call" ? scope.id : `question ${scope.id}`) : "whole run"}`;
    body = <DrillPanel drill={id} scope={scope} />;
  } else if (kind === "evidence") {
    const [part, family] = id.split(":") as [EvidencePart, EvidenceFamily];
    if (EVIDENCE_PARTS.includes(part) && ["compliance", "extraction", "all"].includes(family)) {
      title = presentation.evidence_parts[part].title;
      subtitle = family === "all" ? "Compliance and extraction checks" : presentation.families[family].label;
      body = <EvidencePanel part={part} family={family} />;
    }
  } else if (kind === "error") {
    title = "Diagnostic";
    body = <DiagnosticPanel index={Number(id)} />;
  }
  // Keyed by the whole address so opening another drawer resets which sections start collapsed.
  return <Drawer key={`${panel}|${focus}|${params.get("scope")}`} open onClose={close} title={title} subtitle={subtitle}>{body}</Drawer>;
}
