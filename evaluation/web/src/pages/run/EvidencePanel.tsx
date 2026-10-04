import { ChevronRight } from "lucide-react";
import { Link, useLocation, useParams } from "react-router-dom";
import { Badge, ErrorPanel, Kpi, Loading, SectionTitle, cx } from "../../components/ui";
import { useEvidence, type EvidenceEntry, type EvidenceTurn, type Ratio } from "../../lib/api";
import { at, pct, ratioPct } from "../../lib/format";
import { useMetaApi } from "../../lib/meta";
import { paths, useSource, type ResultNavState } from "../../lib/url";
import { useRunView } from "./context";

export type EvidencePart = "precision" | "recall" | "exact" | "invalid" | "unclear";
export type EvidenceFamily = "compliance" | "extraction" | "all";

export const EVIDENCE_PARTS: EvidencePart[] = ["precision", "recall", "exact", "invalid", "unclear"];

/** The metric each part explains, so the drawer can show its headline number and wording. */
export function evidenceMetricId(part: EvidencePart, family: EvidenceFamily): string {
  if (family === "all") return "diagnostic.evidence_all_questions_precision";
  const names: Record<EvidencePart, string> = {
    precision: "evidence_precision",
    recall: "evidence_recall",
    exact: "evidence_exact_set_match",
    invalid: "invalid_evidence_reference_rate",
    unclear: "native_unclear_outputs",
  };
  return `${family}.${names[part]}`;
}

/** Which checks contribute to the metric (those that are not perfect on it). */
const CONTRIBUTES: Record<EvidencePart, (entry: EvidenceEntry) => boolean> = {
  precision: (entry) => (entry.precision ?? 1) < 1,
  recall: (entry) => (entry.recall ?? 1) < 1,
  exact: (entry) => (entry.exact ?? 1) < 1,
  invalid: (entry) => entry.invalid_reference_count > 0,
  unclear: (entry) => entry.native_verdict === "Unclear",
};

/** The turns responsible for each metric. */
function turnsFor(entry: EvidenceEntry, part: EvidencePart): { kind: "extra" | "missed"; turn: EvidenceTurn }[] {
  const extra = entry.extra.map((turn) => ({ kind: "extra" as const, turn }));
  const missed = entry.missed.map((turn) => ({ kind: "missed" as const, turn }));
  if (part === "precision") return extra;
  if (part === "recall") return missed;
  if (part === "exact") return [...missed, ...extra];
  return [];
}

const KIND = {
  extra: { label: "Extra", border: "border-o-correct" },
  missed: { label: "Missed", border: "border-o-unclear" },
};

function severity(entry: EvidenceEntry, part: EvidencePart): number {
  if (entry.execution_status === "prediction_failure") return 2;  // evidence is zero by convention: show last
  if (part === "precision") return entry.precision ?? 1;
  if (part === "recall") return entry.recall ?? 1;
  if (part === "exact") return -(entry.missed.length + entry.extra.length);
  return 0;
}

function Scores({ entry, part }: { entry: EvidenceEntry; part: EvidencePart }) {
  const value = (v: number | null) => (v === null ? "—" : pct(v * 100));
  if (part === "precision") return <>Precision {value(entry.precision)}</>;
  if (part === "recall") return <>Recall {value(entry.recall)}</>;
  if (part === "exact") return <>Missed {entry.missed.length} · Extra {entry.extra.length}</>;
  if (part === "invalid") return <>{entry.invalid_reference_count} unmatched</>;
  return <>{entry.native_verdict ?? "—"}</>;
}

export function EvidencePanel({ part, family }: { part: EvidencePart; family: EvidenceFamily }) {
  const source = useSource();
  const location = useLocation();
  const { runId = "" } = useParams();
  const { data: run } = useRunView();
  const { metric, presentation, definition } = useMetaApi();
  const evidence = useEvidence(source, runId);
  const metricId = evidenceMetricId(part, family);
  const copy = presentation.evidence_parts[part] as { title: string; what: string; by_rule_title?: string; by_rule_note?: string };

  // The headline is the figure for checks where evidence was expected (the one that says whether the needed
  // turns were found). The all-checks figure also counts checks that expect none, so it is a footnote.
  const overall = metric(run.summary.overall, metricId);
  const breakdowns = definition(metricId)?.breakdowns;
  const expected = family !== "all" && ["precision", "recall", "exact"].includes(part) && breakdowns
    ? (at(run.summary.overall, breakdowns.non_empty_gold) as Ratio | undefined)
    : undefined;
  const none = family !== "all" && breakdowns ? (at(run.summary.overall, breakdowns.empty_gold) as Ratio | undefined) : undefined;
  // Without the wording (an older server), show the plain single figure rather than unlabeled numbers.
  const hasWording = Boolean(presentation.evidence_parts.breakdown_caption && presentation.evidence_parts.breakdown_footnote && presentation.evidence_parts.breakdown_both);
  const hasSplit = hasWording && Boolean(expected && typeof expected === "object" && "denominator" in expected && overall);
  // Recall is about finding the expected turns, so its headline is the "evidence expected" figure. Precision and
  // exact match matter in both kinds of check, so they keep the all-checks headline and show both figures.
  const recallStyle = part === "recall";
  const fill = (template: string, values: Record<string, string | number>) =>
    template.replace(/\{(\w+)\}/g, (_, key: string) => String(values[key] ?? ""));
  const show = (ratio: Ratio) => `${ratioPct(ratio)} (${ratio.numerator} of ${ratio.denominator})`;
  const hasNone = Boolean(none && typeof none === "object" && "denominator" in none);
  const headline = hasSplit && recallStyle ? (expected as Ratio) : overall;
  const headlineCaption = hasSplit && recallStyle
    ? fill(presentation.evidence_parts.breakdown_caption ?? "", { numerator: expected!.numerator, denominator: expected!.denominator })
    : undefined;
  let footnote: string | null = null;
  if (hasSplit && hasNone && overall) {
    footnote = recallStyle
      ? fill(presentation.evidence_parts.breakdown_footnote ?? "", { all: ratioPct(overall), total: overall.denominator, none: (none as Ratio).denominator })
      : fill(presentation.evidence_parts.breakdown_both ?? "", { expected: show(expected as Ratio), none: show(none as Ratio) });
  }

  const entries = (evidence.data?.entries ?? [])
    .filter((entry) => (family === "all" || entry.family === family) && CONTRIBUTES[part](entry))
    .sort((a, b) => severity(a, part) - severity(b, part));

  // Checks with a turn to point at come first. The rest score 0% only through the empty-evidence rule
  // (nothing expected but something cited, or something expected but nothing cited).
  const showsTurns = (entry: EvidenceEntry) => part === "invalid" || part === "unclear" || turnsFor(entry, part).length > 0;
  const withTurns = entries.filter(showsTurns);
  const byRule = entries.filter((entry) => !showsTurns(entry));

  // Turns that show up in several checks of the same call are the ones worth a closer look.
  const tally = new Map<string, { entry: EvidenceEntry; turn: EvidenceTurn; kind: "extra" | "missed"; checks: number }>();
  for (const entry of entries) {
    if (entry.execution_status === "prediction_failure") continue;
    for (const { kind, turn } of turnsFor(entry, part)) {
      const key = `${entry.call_id}|${turn.id}|${kind}`;
      const current = tally.get(key);
      if (current) current.checks += 1;
      else tally.set(key, { entry, turn, kind, checks: 1 });
    }
  }
  const repeated = [...tally.values()].filter((item) => item.checks > 1).sort((a, b) => b.checks - a.checks).slice(0, 5);

  const state: ResultNavState = {
    from: location.pathname + location.search,
    list: entries.map((entry) => [entry.call_id, entry.question_id]),
    listLabel: copy.title,
  };

  return (
    <div className="space-y-5">
      <div className="grid grid-cols-[minmax(0,200px)_1fr] items-start gap-4">
        <Kpi id={metricId} ratio={headline} caption={headlineCaption} />
        <p className="text-[13px] text-ink-2">{copy.what}</p>
      </div>
      {footnote && <p className="text-xs text-ink-3">{footnote}</p>}
      <p className="text-xs text-ink-3">{presentation.evidence_parts.intro}</p>

      {evidence.isPending && <Loading label="Loading evidence" />}
      {evidence.isError && <ErrorPanel error={evidence.error} />}
      {evidence.data && !entries.length && (
        <p className="py-6 text-center text-sm text-ink-3">No checks contribute to this number.</p>
      )}

      {repeated.length > 0 && (
        <section>
          <SectionTitle>Recurring turns</SectionTitle>
          <ul className="space-y-2">
            {repeated.map(({ entry, turn, kind, checks }) => (
              <li key={`${entry.call_id}|${turn.id}|${kind}`} className={cx("rounded-r-md border-l-[3px] bg-surface-2 py-1.5 pl-3 pr-2", KIND[kind].border)}>
                <div className="text-[11px] font-medium text-ink-3">
                  {entry.call_id} · Turn {turn.id} · {turn.speaker} · {KIND[kind].label.toLowerCase()} in {checks} checks
                </div>
                <div className="text-[13px] text-ink">“{turn.text}”</div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {withTurns.length > 0 && (
        <section>
          <SectionTitle>{withTurns.length} {withTurns.length === 1 ? "check" : "checks"}, worst first</SectionTitle>
          <EntryList entries={withTurns} part={part} state={state} />
        </section>
      )}
      {byRule.length > 0 && (
        <section>
          <SectionTitle>{copy.by_rule_title ?? "Other checks"} ({byRule.length})</SectionTitle>
          {copy.by_rule_note && <p className="mb-2 -mt-1 text-xs text-ink-3">{copy.by_rule_note}</p>}
          <EntryList entries={byRule} part={part} state={state} />
        </section>
      )}
    </div>
  );
}

function EntryList({ entries, part, state }: { entries: EvidenceEntry[]; part: EvidencePart; state: ResultNavState }) {
  const source = useSource();
  const { datasetId = "", runId = "" } = useParams();
  return (
    <ul className="space-y-3">
      {entries.map((entry) => {
        const turns = turnsFor(entry, part);
        const failed = entry.execution_status === "prediction_failure";
        return (
          <li key={`${entry.call_id}/${entry.question_id}`} className="rounded-md border border-line">
            <Link to={paths.result(source, datasetId, runId, entry.call_id, entry.question_id)} state={state}
              className="focus-ring group flex items-center gap-3 px-3 py-2 hover:bg-surface-2">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 text-[13px]">
                  <span className="font-medium text-ink">{entry.call_id} · {entry.question_id}</span>
                  <span className="num text-ink-3"><Scores entry={entry} part={part} /></span>
                </div>
                {entry.question && <div className="truncate text-xs text-ink-2">{entry.question}</div>}
              </div>
              {failed && <Badge tone="bad">Prediction failed</Badge>}
              <ChevronRight size={14} className="shrink-0 text-ink-3 opacity-50 group-hover:opacity-100" />
            </Link>
            <div className="space-y-1.5 border-t border-line px-3 py-2">
              {turns.map(({ kind, turn }) => (
                <div key={`${kind}-${turn.id}`} className={cx("rounded-r-md border-l-[3px] bg-surface-2 py-1.5 pl-3 pr-2", KIND[kind].border)}>
                  <div className="text-[11px] font-medium text-ink-3">
                    <span className="text-ink-2">{KIND[kind].label}</span> · Turn {turn.id} · {turn.speaker}
                  </div>
                  <div className="text-[13px] text-ink">“{turn.text}”</div>
                </div>
              ))}
              {part === "invalid" && entry.unresolved_references.length > 0 && (
                <ul className="font-mono text-xs text-ink-2">
                  {entry.unresolved_references.map((reference) => <li key={reference}>{reference}</li>)}
                </ul>
              )}
              {(turns.length === 0 && part !== "invalid") || failed ? (
                <p className="text-xs text-ink-3">{entry.explanation}</p>
              ) : null}
            </div>
          </li>
        );
      })}
    </ul>
  );
}
