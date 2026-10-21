import type { ReactNode } from "react";
import { Tip, cx } from "../../components/ui";
import { pct } from "../../lib/format";
import { useMetaApi } from "../../lib/meta";
import type { SttCounts, SttEntity, SttOp, SttPresentation } from "../../lib/stt";

/** Wording from stt_presentation.json (editable without a restart), with a plain fallback for a missing entry. */
export function useSttWords(): SttPresentation {
  const { stt_presentation } = useMetaApi();
  return stt_presentation as SttPresentation;
}

export function rate(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : pct(value * 100);
}

export function clock(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${(seconds - minutes * 60).toFixed(1).padStart(4, "0")}`;
}

export function fill(template: string | undefined, values: Record<string, string | number>): string {
  return (template ?? "").replace(/\{(\w+)\}/g, (_, key: string) => String(values[key] ?? ""));
}

export function isAvailable(entity: SttEntity): entity is Extract<SttEntity, { reference_words: number }> {
  return !("available" in entity);
}

export function MetricTipStt({ id }: { id: string }) {
  const entry = useSttWords().metrics?.[id];
  if (!entry) return null;
  return <Tip content={{ title: entry.label, what: entry.what, calculation: entry.calculation, important: entry.important }} />;
}

export function SttKpi({ id, value, caption, onClick, label }: {
  id: string; value: string; caption?: ReactNode; onClick?: () => void; label?: string;
}) {
  const entry = useSttWords().metrics?.[id];
  const body = (
    <>
      <div className="num text-[22px] font-semibold leading-7 text-ink">{value}</div>
      {caption && <div className="num text-xs text-ink-3">{caption}</div>}
    </>
  );
  return (
    <div className="min-w-0 rounded-md border border-line bg-surface px-3 py-2">
      <div className="mb-0.5 flex items-center gap-1 text-xs text-ink-2">{label ?? entry?.label ?? id}<MetricTipStt id={id} /></div>
      {onClick ? <button type="button" onClick={onClick} className="drill block w-full text-left">{body}</button> : body}
    </div>
  );
}

export function sdi(counts: Pick<SttCounts, "substitutions" | "deletions" | "insertions">): string {
  return `S ${counts.substitutions} · D ${counts.deletions} · I ${counts.insertions}`;
}

const OP: Record<SttOp | "mixed", { label: string; tone: string }> = {
  equal: { label: "Correct", tone: "bg-tint-correct text-accent-ink" },
  substitute: { label: "Substituted", tone: "bg-tint-wrong text-ink-wrong" },
  delete: { label: "Missing", tone: "bg-tint-wrong text-ink-wrong" },
  insert: { label: "Extra word", tone: "bg-tint-warn text-ink-warn" },
  mixed: { label: "Mixed errors", tone: "bg-tint-wrong text-ink-wrong" },
};

export function OpBadge({ op }: { op: SttOp | "mixed" }) {
  return <span className={cx("inline-flex items-center whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium", OP[op].tone)}>{OP[op].label}</span>;
}

export const opLabel = (op: SttOp | "mixed") => OP[op].label;

const EVENT_LABEL: Record<string, string> = {
  VALUE_CORRECTION: "Correction", ORDER_INVALIDATION: "Cancellation", READBACK: "Read-back", FALSE_START: "False start",
};

/** A readable name for an entity-span group such as value:K01, event:READBACK:linked or term:NRIC. */
export function groupLabel(group: string): string {
  const [kind, name = "", linked] = group.split(":");
  if (kind === "value") return `Value · ${name.replace(/_/g, " ")}`;
  if (kind === "event") return `${EVENT_LABEL[name] ?? name}${linked ? " (linked)" : ""}`;
  if (kind === "term") return `Domain term · ${name}`;
  return group;
}

/** An internal alignment token as a person would say it: the ordinal marker is "the -th/-st/-nd/-rd ending", never "ord". */
export function tokenLabel(token: string | null): string {
  if (token === null || token === "") return "—";
  return token === "ord" ? "(ordinal ending)" : token;
}

export function plainOrDash(token: string | null): string {
  return tokenLabel(token);
}

/** The raw phrase behind an error group, e.g. “13 May” → “13th”, so a token pair is never the only thing shown. */
export function rawPhrase(example: { ref_text: string; hyp_text: string } | undefined): string | null {
  if (!example) return null;
  return `“${example.ref_text || "—"}” → “${example.hyp_text || "—"}”`;
}
