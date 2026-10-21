import { useState, type ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { DataTable, type Column } from "../../components/DataTable";
import { Drawer } from "../../components/Drawer";
import { Page } from "../../components/Page";
import { Badge, Card, Disclosure, ErrorPanel, Loading, SectionTitle, Tip, cx } from "../../components/ui";
import { plural, sgt } from "../../lib/format";
import { sttPaths, useSttRun, useSttRunWords, type SttCounts, type SttEntitySpan, type SttErrorGroup, type SttRun, type SttRunCallRow, type SttWordEntry } from "../../lib/stt";
import { useQueryState } from "../../lib/url";
import { OpBadge, SttKpi, fill, groupLabel, isAvailable, plainOrDash, rate, rawPhrase, sdi, useSttWords } from "./parts";

const OPS = [
  { op: "substitute", metric: "substitutions", key: "substitutions", title: "Substitutions" },
  { op: "delete", metric: "deletions", key: "deletions", title: "Deletions" },
  { op: "insert", metric: "insertions", key: "insertions", title: "Insertions" },
] as const;

function Coverage({ run }: { run: SttRun }) {
  const words = useSttWords();
  const { summary } = run;
  const lines: string[] = [fill(words.coverage?.calls, { scored: summary.calls_scored, total: run.dataset_call_ids.length || "?" })];
  if (!isAvailable(summary.entity_wer)) {
    lines.push(fill(words.coverage?.entity_unavailable, { reason: summary.entity_wer.reason }));
  } else {
    const terms = summary.entity_wer.domain_terms;
    lines.push(terms.reviewed && terms.terms ? fill(words.coverage?.terms_some, { count: terms.terms }) : words.coverage?.terms_none ?? "");
  }
  if (summary.calls_not_completed.length) lines.push(`Not scored (the job did not complete): ${summary.calls_not_completed.join(", ")}.`);
  return (
    <Card className="mb-4 p-3">
      <ul className="space-y-0.5 text-[13px] text-ink-2">
        {lines.map((line) => <li key={line}>{line}</li>)}
        <li className="text-ink-3">Text normalisation version {run.versions.normalizer}: {words.normaliser_note}</li>
      </ul>
    </Card>
  );
}

type Indexed = { group: SttErrorGroup; index: number };

function GroupRows({ groups, onOpen, empty }: { groups: Indexed[]; onOpen: (index: number) => void; empty: string }) {
  if (!groups.length) return <p className="px-3 py-4 text-center text-sm text-ink-3">{empty}</p>;
  return (
    <ul className="divide-y divide-line">
      {groups.map(({ group, index }) => (
        <li key={index}>
          <button type="button" onClick={() => onOpen(index)}
            className="focus-ring flex w-full items-center gap-3 rounded px-2 py-1.5 text-left text-[13px] hover:bg-surface-2">
            <span className="w-24 shrink-0"><OpBadge op={group.op} /></span>
            <span className="min-w-0 flex-1">
              <span className="block truncate">
                <span className="text-ink-2">{plainOrDash(group.ref)}</span>
                <span className="mx-1.5 text-ink-3">→</span>
                <span className="font-medium text-ink">{plainOrDash(group.hyp)}</span>
              </span>
              {rawPhrase(group.examples[0]) && <span className="block truncate text-xs text-ink-3">as written {rawPhrase(group.examples[0])}</span>}
            </span>
            {group.entity_count > 0 && <span className="shrink-0 text-xs text-ink-wrong">{group.entity_count} in entity spans</span>}
            <span className="num w-8 shrink-0 text-right font-semibold text-ink">{group.count}</span>
          </button>
        </li>
      ))}
    </ul>
  );
}

function ErrorDrawer({ run, panel, close, open }: { run: SttRun; panel: string | null; close: () => void; open: (panel: string) => void }) {
  const { datasetId = "" } = useParams();
  if (!panel) return <Drawer open={false} onClose={close} title="">{null}</Drawer>;
  if (panel.startsWith("kpi:")) return <KpiDrawer run={run} metric={panel.slice(4)} close={close} open={open} />;
  if (panel.startsWith("grp:")) return <EntityGroupDrawer run={run} group={panel.slice(4)} close={close} />;
  const indexed: Indexed[] = run.errors.map((group, index) => ({ group, index }));
  if (panel.startsWith("op:")) {
    const op = panel.slice(3);
    const rows = indexed.filter(({ group }) => group.op === op);
    const total = rows.reduce((sum, { group }) => sum + group.count, 0);
    return (
      <Drawer open onClose={close} title={OPS.find((entry) => entry.op === op)?.title ?? "Errors"}
        subtitle={`${plural(total, "error")} in ${plural(rows.length, "kind")}, entity-related first`}>
        <GroupRows groups={rows} onOpen={(index) => open(`err:${index}`)} empty="No errors of this kind." />
      </Drawer>
    );
  }
  const group = run.errors[Number(panel.slice(4))];
  if (!group) return <Drawer open onClose={close} title="Error"><p className="text-sm text-ink-3">Not found.</p></Drawer>;
  const explanation = group.op === "substitute" ? "The reference word was replaced by another." :
    group.op === "delete" ? "The reference word is missing from Transcribe's output." : "Transcribe added this word.";
  return (
    <Drawer open onClose={close} title={<span>{plainOrDash(group.ref)} <span className="text-ink-3">→</span> {plainOrDash(group.hyp)}</span>}
      subtitle={`${plural(group.count, "occurrence")}${group.entity_count ? ` · ${group.entity_count} inside entity spans` : ""}`}>
      <div className="flex items-center gap-2 text-[13px]"><OpBadge op={group.op} /><span className="text-ink-2">{explanation}</span></div>
      {group.examples.length > 0 && (
        <div>
          <SectionTitle>As written (raw forms)</SectionTitle>
          <ul className="space-y-1 text-[13px]">
            {group.examples.map((example, index) => (
              <li key={index} className="rounded border border-line px-2 py-1">
                <span className="text-ink-2">said: </span>{example.ref_text || "—"}
                <span className="mx-2 text-ink-3">|</span>
                <span className="text-ink-2">Transcribe: </span>{example.hyp_text || "—"}
              </li>
            ))}
          </ul>
        </div>
      )}
      <div>
        <SectionTitle>Where it happens</SectionTitle>
        <ul className="space-y-2">
          {group.occurrences.map((occurrence) => (
            <li key={`${occurrence.call_id}-${occurrence.unit}`}>
              <Link to={sttPaths.call(datasetId, run.run_id, occurrence.call_id, occurrence.unit)}
                className="focus-ring block rounded-md border border-line px-3 py-2 text-[13px] hover:bg-surface-2">
                <div className="mb-0.5 flex items-center gap-2 text-xs text-ink-3">
                  <span className="font-medium text-ink-2">{occurrence.call_id}</span> turn {occurrence.turn_id}
                  {occurrence.entity && <Badge tone="warn" icon={false}>entity span</Badge>}
                </div>
                <span className="text-ink-3">…{occurrence.context.before} </span>
                <span className="rounded bg-tint-wrong px-1 font-medium text-ink-wrong">{plainOrDash(group.ref)} → {plainOrDash(group.hyp)}</span>
                <span className="text-ink-3"> {occurrence.context.after}…</span>
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </Drawer>
  );
}

const ROLE: Record<string, string> = { EMPLOYEE: "Employee", CLIENT: "Client" };
const role = (value: string | null | undefined) => (value ? ROLE[value] ?? value : "no role");

/** A horizontal bar for a value, scaled to the largest value shown. */
function Bar({ value, max, tone = "bg-o-wrong" }: { value: number; max: number; tone?: string }) {
  return (
    <span className="block h-1.5 w-full rounded-full bg-surface-2">
      <span className={cx("block h-1.5 rounded-full", tone)} style={{ width: `${max > 0 && value > 0 ? Math.max(2, (value / max) * 100) : 0}%` }} />
    </span>
  );
}

function CallLink({ run, callId, unit, children }: { run: SttRun; callId: string; unit?: number; children: ReactNode }) {
  const { datasetId = "" } = useParams();
  return (
    <Link to={sttPaths.call(datasetId, run.run_id, callId, unit)} className="focus-ring block rounded-md px-2 py-1.5 text-[13px] hover:bg-surface-2">{children}</Link>
  );
}

function WordList({ run, entries, empty, detail }: { run: SttRun; entries: SttWordEntry[] | undefined; empty: string; detail: (entry: SttWordEntry) => ReactNode }) {
  if (!entries) return <Loading label="Reading the calls" />;
  if (!entries.length) return <p className="text-xs text-ink-3">{empty}</p>;
  return (
    <ul className="divide-y divide-line">
      {entries.map((entry) => (
        <li key={`${entry.call_id}-${entry.unit}`}>
          <CallLink run={run} callId={entry.call_id} unit={entry.unit}>
            <div className="flex flex-wrap items-baseline gap-x-2">
              <span className="font-medium text-ink">{entry.ref_text || <i className="text-ink-3">(not in Gold)</i>}</span>
              {entry.hyp_text !== entry.ref_text && <span className="text-ink-3">→ <span className="text-ink-2">{entry.hyp_text || "(not transcribed)"}</span></span>}
              {entry.errors > 0 && <Badge tone="bad" icon={false}>error</Badge>}
              <span className="ml-auto shrink-0 text-xs text-ink-3">{entry.call_id} · turn {entry.turn_id}</span>
            </div>
            <div className="text-xs text-ink-2">{detail(entry)}</div>
          </CallLink>
        </li>
      ))}
    </ul>
  );
}

function WordsError({ error }: { error: unknown }) {
  return (
    <div>
      <ErrorPanel error={error} />
      <p className="mt-1 text-xs text-ink-3">If this says "Not found", the dashboard server was started before this view existed: stop it and start it again, then reload.</p>
    </div>
  );
}

const field = "focus-ring rounded-md border border-line bg-surface px-2 py-1 text-xs text-ink";

/** Words from every call, with a search box, a confidence ceiling, "mistakes only", a sort order and a top-N cut, so a long list stays usable. */
function WordExplorer({ run, entries, empty, detail, confidence = false, title, aside }: {
  run: SttRun; entries: SttWordEntry[] | undefined; empty: string; detail: (entry: SttWordEntry) => ReactNode; confidence?: boolean;
  title: string; aside?: string;
}) {
  const [search, setSearch] = useState("");
  const [below, setBelow] = useState(0.5);
  const [mistakes, setMistakes] = useState(false);
  const [order, setOrder] = useState<"lowest" | "call">(confidence ? "lowest" : "call");
  const [top, setTop] = useState(25);
  if (!entries) return <Loading label="Reading the calls" />;
  const needle = search.trim().toLowerCase();
  let rows = entries.filter((entry) => (!confidence || (entry.min_confidence ?? 1) < below) && (!mistakes || entry.errors > 0)
    && (!needle || [entry.ref_text, entry.hyp_text, entry.call_id, ...entry.items.map((item) => item.content)].join(" ").toLowerCase().includes(needle)));
  rows = [...rows].sort((a, b) => (order === "lowest"
    ? (a.min_confidence ?? 1) - (b.min_confidence ?? 1)
    : a.call_id.localeCompare(b.call_id) || a.turn_id - b.turn_id || a.unit - b.unit));
  const shown = top ? rows.slice(0, top) : rows;
  return (
    <div>
      <SectionTitle aside={aside ?? String(rows.length)}>{title}</SectionTitle>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <input type="search" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search a word or call" aria-label="Search words"
          className={cx(field, "min-w-[150px] flex-1")} />
        {confidence && (
          <label className="flex items-center gap-1 text-xs text-ink-2">Below
            <select value={below} onChange={(event) => setBelow(Number(event.target.value))} className={field} aria-label="Confidence below">
              <option value={0.5}>0.5</option><option value={0.7}>0.7</option><option value={0.9}>0.9</option>
            </select>
          </label>
        )}
        <label className="flex items-center gap-1 text-xs text-ink-2">Sort
          <select value={order} onChange={(event) => setOrder(event.target.value as "lowest" | "call")} className={field} aria-label="Sort">
            {confidence && <option value="lowest">Lowest confidence</option>}<option value="call">Call order</option>
          </select>
        </label>
        <label className="flex items-center gap-1 text-xs text-ink-2">Show
          <select value={top} onChange={(event) => setTop(Number(event.target.value))} className={field} aria-label="How many">
            <option value={10}>Top 10</option><option value={25}>Top 25</option><option value={50}>Top 50</option><option value={0}>All</option>
          </select>
        </label>
        <label className="flex items-center gap-1 text-xs text-ink-2"><input type="checkbox" checked={mistakes} onChange={(event) => setMistakes(event.target.checked)} />Mistakes only</label>
      </div>
      <p className="mb-1 text-xs text-ink-3" data-testid="explorer-count">Showing {shown.length} of {rows.length}{rows.length !== entries.length ? ` (${entries.length} before filters)` : ""}</p>
      <WordList run={run} entries={shown} empty={empty} detail={detail} />
    </div>
  );
}

function CallBars({ run, rows, value, shown, extra, bars = true }: {
  run: SttRun; rows: SttRunCallRow[]; value: (row: SttRunCallRow) => number | null; shown: (row: SttRunCallRow) => string; extra?: (row: SttRunCallRow) => ReactNode;
  bars?: boolean;
}) {
  const max = Math.max(0, ...rows.map((row) => value(row) ?? 0));
  return (
    <ul className="space-y-0.5">
      {rows.map((row) => (
        <li key={row.call_id}>
          <CallLink run={run} callId={row.call_id}>
            <div className="flex items-center gap-2">
              <span className="w-20 shrink-0 font-medium text-ink">{row.call_id}</span>
              <span className="num w-14 shrink-0 text-right font-semibold text-ink">{shown(row)}</span>
              <span className="min-w-[60px] flex-1">{bars && <Bar value={value(row) ?? 0} max={max} />}</span>
              {extra && <span className="num shrink-0 text-xs text-ink-3">{extra(row)}</span>}
            </div>
          </CallLink>
        </li>
      ))}
    </ul>
  );
}

/** Drill-downs behind the four headline cards. Everything shown is read from the exported bundle; nothing is re-scored. */
/** Every annotated occurrence of one kind of entity (an identifier, an amount, a date, a correction...), across the run's calls. */
function EntityGroupDrawer({ run, group, close }: { run: SttRun; group: string; close: () => void }) {
  const words = useSttWords();
  const query = useSttRunWords(run.run_id, true);
  const counts = isAvailable(run.summary.entity_wer) ? run.summary.entity_wer.groups[group] : undefined;
  const spans: SttEntitySpan[] = (query.data?.entity_spans ?? []).filter((span) => span.group === group);
  return (
    <Drawer open onClose={close} title={groupLabel(group)}
      subtitle={counts ? `${counts.reference_words} words · ${plural(counts.errors, "error")} · ${rate(counts.wer)}` : undefined}>
      <p className="text-xs text-ink-3">{words.drills?.entity_group_help}</p>
      {query.isError ? <WordsError error={query.error} /> : !query.data ? <Loading label="Reading the calls" /> : !spans.length ? (
        <p className="text-xs text-ink-3">No occurrences of this kind.</p>
      ) : (
        <ul className="divide-y divide-line">
          {spans.map((span, index) => (
            <li key={`${span.call_id}-${span.turn_id}-${index}`}>
              <CallLink run={run} callId={span.call_id} unit={span.unit ?? undefined}>
                <div className="flex flex-wrap items-baseline gap-x-2">
                  <span className="font-medium text-ink">{span.text}</span>
                  {span.errors > 0 ? <Badge tone="bad" icon={false}>{plural(span.errors, "error")}</Badge> : <Badge tone="good" icon={false}>correct</Badge>}
                  <span className="ml-auto shrink-0 text-xs text-ink-3">{span.call_id} · turn {span.turn_id}</span>
                </div>
                <div className="text-xs text-ink-2">Transcribe: {span.hypothesis_tokens.join(" ") || "—"}</div>
              </CallLink>
            </li>
          ))}
        </ul>
      )}
    </Drawer>
  );
}

function KpiDrawer({ run, metric, close, open }: { run: SttRun; metric: string; close: () => void; open: (panel: string) => void }) {
  const words = useSttWords();
  const drills = words.drills ?? {};
  const runWords = useSttRunWords(run.run_id, metric === "conf" || metric === "speaker");
  const { summary } = run;
  const label = (id: string) => words.metrics?.[id]?.label ?? id;
  const calls = [...run.calls];
  const indexed = run.errors.map((group, index) => ({ group, index }));

  if (metric === "wer") {
    const wer = summary.normalised_wer;
    calls.sort((a, b) => (b.wer.wer ?? 0) - (a.wer.wer ?? 0));
    return (
      <Drawer open onClose={close} title={label("normalised_wer")} subtitle={`${rate(wer.wer)} · ${plural(wer.errors, "error")} in ${wer.reference_words} words`}>
        <div className="grid grid-cols-3 gap-2">
          {OPS.map(({ op, key, title }) => (
            <button key={op} type="button" disabled={!wer[key]} onClick={() => open(`op:${op}`)}
              className="focus-ring rounded-md border border-line px-3 py-2 text-left enabled:hover:bg-surface-2 disabled:opacity-60">
              <div className="text-xs text-ink-2">{title}</div><div className="num text-lg font-semibold text-ink">{wer[key]}</div>
            </button>
          ))}
        </div>
        <div>
          <SectionTitle>{drills.wer_calls ?? "Calls, worst first"}</SectionTitle>
          <CallBars run={run} rows={calls} value={(row) => row.wer.wer} shown={(row) => rate(row.wer.wer)} extra={(row) => `${row.wer.errors} · ${sdi(row.wer)}`} />
        </div>
        <div>
          <SectionTitle>Most frequent errors</SectionTitle>
          <GroupRows groups={indexed.slice(0, 8)} onOpen={(index) => open(`err:${index}`)} empty="No errors." />
        </div>
      </Drawer>
    );
  }

  if (metric === "entity") {
    const entity = summary.entity_wer;
    const entry = words.metrics?.entity_wer;
    if (!isAvailable(entity)) {
      return (
        <Drawer open onClose={close} title={label("entity_wer")} subtitle="Not available for this run">
          <p className="text-[13px] text-ink">Reason: {entity.reason}.</p>
          <p className="text-[13px] text-ink-2">{drills.entity_unavailable_help}</p>
          {entry && <div className="rounded-md border border-line p-3 text-xs text-ink-2"><p className="mb-1">{entry.calculation}</p><p className="text-ink-3">{entry.important}</p></div>}
        </Drawer>
      );
    }
    const groups = Object.entries(entity.groups).sort((a, b) => b[1].errors - a[1].errors);
    const entityErrors = indexed.filter(({ group }) => group.entity_count > 0);
    calls.sort((a, b) => (b.entity_wer?.wer ?? -1) - (a.entity_wer?.wer ?? -1));
    return (
      <Drawer open onClose={close} title={label("entity_wer")} subtitle={`${rate(entity.wer)} · ${plural(entity.errors, "error")} in ${entity.reference_words} entity words`}>
        <div>
          <SectionTitle aside={`${groups.length} kinds`}>{drills.entity_kinds_title ?? "Entity words by kind"}</SectionTitle>
          <p className="mb-1 text-xs text-ink-3">{drills.entity_kinds_help}</p>
          <ul className="divide-y divide-line text-[13px]">
            {groups.map(([group, counts]) => (
              <li key={group}>
                <button type="button" onClick={() => open(`grp:${group}`)} title="Show every occurrence"
                  className="focus-ring flex w-full items-baseline gap-2 rounded px-2 py-1.5 text-left hover:bg-surface-2">
                  <span className="min-w-0 flex-1 truncate font-medium text-ink">{groupLabel(group)}</span>
                  <span className="num text-xs text-ink-3">{counts.reference_words} words · {counts.errors} {counts.errors === 1 ? "error" : "errors"}</span>
                  <span className="num w-14 text-right font-semibold text-ink">{rate(counts.wer)}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <SectionTitle>Calls, worst first</SectionTitle>
          <CallBars run={run} rows={calls} value={(row) => row.entity_wer?.wer ?? null} shown={(row) => (row.entity_wer ? rate(row.entity_wer.wer) : "—")} />
        </div>
        <div>
          <SectionTitle aside={`${entityErrors.length} kinds`}>Errors inside entity spans</SectionTitle>
          <GroupRows groups={entityErrors} onOpen={(index) => open(`err:${index}`)} empty="No errors inside entity spans." />
        </div>
      </Drawer>
    );
  }

  if (metric === "conf") {
    const confidence = summary.average_confidence;
    calls.sort((a, b) => (a.mean_confidence ?? 1) - (b.mean_confidence ?? 1));
    const buckets = runWords.data?.confidence_buckets;
    return (
      <Drawer open onClose={close} title={label("average_confidence")}
        subtitle={`${confidence.mean ?? "—"} over ${confidence.items} words · matched ${confidence.mean_correct_items ?? "—"} · others ${confidence.mean_other_items ?? "—"}`}>
        <div>
          <SectionTitle>{drills.confidence_buckets ?? "How confidence relates to mistakes"}</SectionTitle>
          {runWords.isError ? <WordsError error={runWords.error} /> : !buckets ? <Loading label="Reading the calls" /> : (
            <table className="w-full text-[13px]" data-testid="confidence-buckets">
              <thead><tr className="text-left text-[11px] text-ink-3"><th className="py-1 font-medium">Confidence</th><th className="text-right font-medium">Matched</th><th className="text-right font-medium">Other</th><th className="pl-3 font-medium">Share that were errors</th></tr></thead>
              <tbody>
                {buckets.map((bucket) => {
                  const total = bucket.matched + bucket.other;
                  const share = total ? bucket.other / total : 0;
                  return (
                    <tr key={bucket.from} className="border-t border-line">
                      <td className="num py-1">{bucket.from.toFixed(1)}–{bucket.to.toFixed(1)}</td>
                      <td className="num text-right">{bucket.matched}</td>
                      <td className="num text-right">{bucket.other}</td>
                      <td className="pl-3"><div className="flex items-center gap-2"><span className="w-full"><Bar value={share} max={1} tone="bg-o-unclear" /></span><span className="num w-12 shrink-0 text-right text-xs text-ink-2">{total ? rate(share) : "—"}</span></div></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          <p className="mt-1 text-xs text-ink-3">{drills.confidence_buckets_help}</p>
        </div>
        <div>
          <SectionTitle>Calls, least confident first</SectionTitle>
          <CallBars run={run} rows={calls} value={(row) => (row.mean_confidence === null ? null : 1 - row.mean_confidence)} shown={(row) => String(row.mean_confidence ?? "—")} bars={false} />
        </div>
        {runWords.isError ? <WordsError error={runWords.error} /> : (
          <WordExplorer run={run} entries={runWords.data?.low_confidence} confidence title={drills.low_confidence ?? "Words Transcribe was unsure of"}
            empty="No words match these filters."
            detail={(entry) => <>lowest confidence <span className="num font-medium text-ink-warn">{entry.min_confidence ?? "—"}</span>{entry.items.length > 1 && ` (${entry.items.map((item) => `${item.content} ${item.confidence ?? "—"}`).join(", ")})`}</>} />
        )}
      </Drawer>
    );
  }

  const speaker = summary.speaker_attribution;
  calls.sort((a, b) => (a.speaker.accuracy ?? 1) - (b.speaker.accuracy ?? 1));
  return (
    <Drawer open onClose={close} title={label("speaker_attribution")}
      subtitle={`${rate(speaker.accuracy)} · ${speaker.correct_items} of ${speaker.eligible_items} words`}>
      <div>
        <SectionTitle>Calls, lowest first</SectionTitle>
        <CallBars run={run} rows={calls} value={(row) => (row.speaker.accuracy === null ? null : 1 - row.speaker.accuracy)} shown={(row) => rate(row.speaker.accuracy)}
          extra={(row) => <span className="font-mono">{Object.entries(row.speaker.label_mapping ?? {}).map(([name, mapped]) => `${name} → ${role(mapped)}`).join(", ")}</span>} />
        <p className="mt-1 text-xs text-ink-3">The bar shows the share of words given to the wrong speaker.</p>
      </div>
      {runWords.isError ? <WordsError error={runWords.error} /> : (
        <WordExplorer run={run} entries={runWords.data?.speaker_mismatch} title={drills.speaker_mismatch ?? "Words given to the wrong speaker"}
          empty="Every matched word went to the right speaker."
          detail={(entry) => <>Gold: <span className="font-medium text-ink">{role(entry.ref_role)}</span> · Transcribe: {[...new Set(entry.items.map((item) => `${item.speaker ?? "?"} → ${role(item.mapped_role)}`))].join(", ")}</>} />
      )}
    </Drawer>
  );
}

type GroupRow = [string, SttCounts];

export default function SttRunPage() {
  const { datasetId = "", runId = "" } = useParams();
  const navigate = useNavigate();
  const words = useSttWords();
  const query = useSttRun(runId);
  const [params, setParams] = useQueryState();
  const filter = params.get("ef") ?? "all";
  const showAll = params.get("all") === "1";
  const open = (panel: string) => setParams({ panel });
  const close = () => setParams({ panel: null });
  const run = query.data;

  const crumbs = [
    { label: "Datasets", to: sttPaths.home() },
    { label: run?.dataset_id.replace(/_/g, " ") ?? datasetId, to: sttPaths.dataset(datasetId) },
    { label: run ? sgt(run.collected_at, { seconds: false }) : runId },
  ];
  if (query.isPending) return <Page section="stt" crumbs={crumbs}><Loading /></Page>;
  if (query.isError || !run) return <Page section="stt" crumbs={crumbs}><ErrorPanel error={query.error} /></Page>;

  const { summary } = run;
  const wer = summary.normalised_wer;
  const entity = summary.entity_wer;
  const confidence = summary.average_confidence;
  const speaker = summary.speaker_attribution;
  const all: Indexed[] = run.errors.map((group, index) => ({ group, index }));
  const shown = all.filter(({ group }) => filter === "all" || (filter === "entity" ? group.entity_count > 0 : group.op === filter));
  const visible = showAll ? shown : shown.slice(0, 12);

  const callColumns: Column<SttRunCallRow>[] = [
    { key: "call", header: "Call", sort: (row) => row.call_id, cell: (row) => <span className="font-medium text-ink">{row.call_id}</span> },
    { key: "wer", header: "Normalised WER", align: "right", sort: (row) => row.wer.wer, cell: (row) => <span className="font-medium text-ink">{rate(row.wer.wer)}</span> },
    { key: "errors", header: "Errors", align: "right", sort: (row) => row.wer.errors, cell: (row) => <span className="num text-xs text-ink-2">{row.wer.errors} ({sdi(row.wer)})</span> },
    { key: "entity", header: "Entity WER", align: "right", sort: (row) => row.entity_wer?.wer ?? null, cell: (row) => (row.entity_wer ? rate(row.entity_wer.wer) : <span className="text-ink-3">—</span>) },
    { key: "speaker", header: "Speaker", align: "right", sort: (row) => row.speaker.accuracy, cell: (row) => rate(row.speaker.accuracy) },
    { key: "conf", header: "Avg. confidence", align: "right", sort: (row) => row.mean_confidence, cell: (row) => <span className="num">{row.mean_confidence ?? "—"}</span> },
  ];
  const groupColumns: Column<GroupRow>[] = [
    { key: "group", header: "Part of the call", sort: ([group]) => group, cell: ([group]) => <span className="font-medium text-ink">{groupLabel(group)}</span> },
    { key: "words", header: "Words", align: "right", sort: ([, c]) => c.reference_words, cell: ([, c]) => <span className="num">{c.reference_words}</span> },
    { key: "errors", header: "Errors", align: "right", sort: ([, c]) => c.errors, cell: ([, c]) => <span className="num text-xs text-ink-2">{c.errors === 0 ? "none" : `${c.errors} (${sdi(c)})`}</span> },
    { key: "wer", header: "WER", align: "right", sort: ([, c]) => c.wer, cell: ([, c]) => <span className="font-medium text-ink">{rate(c.wer)}</span> },
  ];

  return (
    <Page section="stt" crumbs={crumbs}>
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h1 className="text-xl font-semibold text-ink">{sgt(run.collected_at)}</h1>
        {run.dataset_call_ids.length > summary.calls_scored && (
          <Badge tone="neutral" icon={false}>Partial: {summary.calls_scored} of {run.dataset_call_ids.length} calls</Badge>
        )}
        <Badge tone="neutral" icon={false}>{run.transcribe_job.vocabulary_name ? `Vocabulary: ${run.transcribe_job.vocabulary_name}` : "No custom vocabulary"}</Badge>
        <span className="font-mono text-xs text-ink-3">{run.run_id}</span>
      </div>
      <Coverage run={run} />

      <div className="mb-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
        <SttKpi id="normalised_wer" value={rate(wer.wer)} caption={`${wer.errors} errors in ${wer.reference_words} words`} onClick={() => open("kpi:wer")} />
        {isAvailable(entity)
          ? <SttKpi id="entity_wer" value={rate(entity.wer)} caption={`${entity.errors} errors in ${entity.reference_words} entity words`} onClick={() => open("kpi:entity")} />
          : <SttKpi id="entity_wer" value="Not available" caption={entity.reason} onClick={() => open("kpi:entity")} />}
        <SttKpi id="average_confidence" value={confidence.mean === null ? "—" : String(confidence.mean)}
          caption={`correct words ${confidence.mean_correct_items ?? "—"} · others ${confidence.mean_other_items ?? "—"}`} onClick={() => open("kpi:conf")} />
        <SttKpi id="speaker_attribution" value={rate(speaker.accuracy)} caption={`${speaker.correct_items} of ${speaker.eligible_items} words`} onClick={() => open("kpi:speaker")} />
      </div>

      <div className="mb-4 grid grid-cols-3 gap-2">
        {OPS.map(({ op, metric, key }) => {
          const entry = words.metrics?.[metric];
          return (
            <div key={op} className="rounded-md border border-line bg-surface px-3 py-2">
              <div className="mb-0.5 flex items-center gap-1 text-xs text-ink-2">
                {entry?.label ?? op}
                {entry && <Tip content={{ title: entry.label, what: entry.what, calculation: entry.calculation, important: entry.important }} />}
              </div>
              <button type="button" disabled={!wer[key]} onClick={() => open(`op:${op}`)} title="Show these errors"
                className={cx("num text-[22px] font-semibold leading-7 text-ink", wer[key] ? "drill" : "cursor-default opacity-60")}>{wer[key]}</button>
            </div>
          );
        })}
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-4">
          <Card>
            <h2 className="border-b border-line px-3 py-2 text-sm font-semibold text-ink">Calls</h2>
            <DataTable id="sttcalls" columns={callColumns} rows={run.calls} rowKey={(row) => row.call_id}
              rowTitle="Inspect this call" onRowClick={(row) => navigate(sttPaths.call(datasetId, runId, row.call_id))} />
          </Card>
          {isAvailable(entity) && (
            <Card>
              <h2 className="border-b border-line px-3 py-2 text-sm font-semibold text-ink">{words.drills?.entity_kinds_title ?? "Entity words by kind"}</h2>
              <p className="border-b border-line px-3 py-1.5 text-xs text-ink-3">{words.drills?.entity_kinds_help}</p>
              <DataTable id="sttgroups" columns={groupColumns} rows={Object.entries(entity.groups) as GroupRow[]} rowKey={([group]) => group}
                rowTitle="Show every occurrence" onRowClick={([group]) => open(`grp:${group}`)} />
            </Card>
          )}
        </div>
        <Card className="min-w-0 p-3">
          <SectionTitle aside={`${shown.length} kinds`}>Most frequent errors</SectionTitle>
          <div className="mb-2 flex flex-wrap gap-1 text-xs">
            {[["all", "All"], ["entity", "In entity spans"], ["substitute", "Substituted"], ["delete", "Missing"], ["insert", "Extra"]].map(([id, label]) => (
              <button key={id} type="button" aria-pressed={filter === id} onClick={() => setParams({ ef: id === "all" ? null : id }, { replace: true })}
                className={cx("focus-ring rounded-full border px-2.5 py-0.5", filter === id ? "border-accent bg-accent text-white" : "border-line text-ink-2 hover:text-ink")}>
                {label}
              </button>
            ))}
          </div>
          <GroupRows groups={visible} onOpen={(index) => open(`err:${index}`)} empty="No errors match this filter." />
          {shown.length > visible.length && (
            <button type="button" onClick={() => setParams({ all: "1" }, { replace: true })} className="mt-2 text-xs font-medium text-accent-ink hover:underline">
              Show all {shown.length}
            </button>
          )}
          <p className="mt-3 text-xs text-ink-3">
            Errors inside entity spans come first: those are the words and numbers that matter for the checks. Click one to see every place it happens.
          </p>
        </Card>
      </div>

      <div className="mt-4 space-y-3">
        <Disclosure title="Forms left unchanged by normalisation">
          {Object.values(run.unnormalised).every((categories) => !Object.keys(categories).length) ? <p className="text-xs text-ink-3">None.</p> : (
            <div className="space-y-2 text-xs text-ink-2">
              {Object.entries(run.unnormalised).flatMap(([side, categories]) => Object.entries(categories).map(([category, info]) => (
                <div key={`${side}-${category}`}>
                  <span className="font-medium text-ink">{side} · {category.replace(/_/g, " ")}</span> ({info.count}): {Object.keys(info.examples).join(", ")}
                </div>
              )))}
            </div>
          )}
        </Disclosure>
        <Disclosure title="Run details">
          <dl className="grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
            {([
              ["Dataset", run.dataset_id], ["Render", run.render_id ?? "—"], ["Collected", sgt(run.collected_at)], ["Scored", sgt(run.scored_at)],
              ["Language", run.transcribe_job.language_code ?? "—"], ["Transcribe settings", JSON.stringify(run.transcribe_job.settings)],
              ["Custom vocabulary", run.transcribe_job.vocabulary_name ?? "none"], ["Custom language model", run.transcribe_job.language_model_name ?? "none"],
              ["Handler commit", `${run.provenance.handler?.git_commit?.slice(0, 8) ?? "—"}${run.provenance.handler?.uncommitted_changes ? " (uncommitted edits)" : ""}`],
              ["fyp-be commit", `${run.provenance.fyp_be_commit?.slice(0, 8) ?? "—"}${run.provenance.fyp_be_uncommitted_changes ? " (+ uncommitted changes)" : ""}`],
              ["Normaliser / jiwer", `${run.versions.normalizer} / ${run.versions.jiwer}`], ["Region", run.provenance.region ?? "—"],
            ] as [string, string][]).map(([label, value]) => (
              <div key={label} className="flex gap-2"><dt className="w-36 shrink-0 text-ink-3">{label}</dt><dd className="font-mono text-ink-2">{value}</dd></div>
            ))}
          </dl>
        </Disclosure>
      </div>
      <ErrorDrawer run={run} panel={params.get("panel")} close={close} open={open} />
    </Page>
  );
}
