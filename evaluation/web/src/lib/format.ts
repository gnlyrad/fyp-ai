import type { Ratio, Row, SummaryNode } from "./api";

/** Same rule as the evaluator's format_percentage: whole numbers drop the decimal. */
export function pct(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Math.abs(value - Math.round(value)) < 1e-9 ? `${Math.round(value)}%` : `${value.toFixed(1)}%`;
}

export function ratioPct(ratio: Ratio | null | undefined): string {
  return ratio && ratio.denominator ? pct(ratio.percentage) : "—";
}

export function fraction(ratio: Ratio | null | undefined): string {
  return ratio ? `${trim(ratio.numerator)} / ${trim(ratio.denominator)}` : "";
}

function trim(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

export function signedPp(value: number | null): string {
  if (value === null) return "—";
  if (Math.abs(value) < 0.005) return "0 pp";
  return `${value > 0 ? "+" : "−"}${Math.abs(value).toFixed(1).replace(/\.0$/, "")} pp`;
}

const SGT_PARTS = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Asia/Singapore",
  day: "2-digit",
  month: "short",
  year: "numeric",
  hour: "numeric",
  minute: "2-digit",
  second: "2-digit",
  hour12: true,
});

/** UTC timestamps are stored; every user-facing time is Singapore time. */
export function sgt(value: string | null | undefined, { seconds = true, year = true } = {}): string {
  if (!value) return "Unknown time";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const part = Object.fromEntries(SGT_PARTS.formatToParts(date).map((p) => [p.type, p.value]));
  const time = `${part.hour}:${part.minute}${seconds ? `:${part.second}` : ""} ${String(part.dayPeriod).toUpperCase()}`;
  return `${part.day} ${part.month}${year ? ` ${part.year}` : ""} · ${time} SGT`;
}

/** Read a dotted summary path, like the evaluator's nested(). */
export function at(node: SummaryNode | null | undefined, path: string): any {
  return path.split(".").reduce<any>((value, key) => (value && typeof value === "object" ? value[key] : undefined), node);
}

export function count(node: SummaryNode | null | undefined, path: string): number {
  const value = at(node, path);
  return typeof value === "number" ? value : 0;
}

export function matches(row: Row, filter: Record<string, string>): boolean {
  return Object.entries(filter).every(([key, value]) => (row as Record<string, unknown>)[key] === value);
}

export function turnList(turns: number[]): string {
  if (!turns.length) return "none";
  const sorted = [...turns].sort((a, b) => a - b);
  return `${sorted.length === 1 ? "Turn" : "Turns"} ${sorted.join(", ")}`;
}

export function shortHash(value: string | null | undefined, length = 7): string {
  return value ? value.slice(0, length) : "—";
}

export function plural(n: number, word: string, many = `${word}s`): string {
  return `${n} ${n === 1 ? word : many}`;
}

export function duration(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "—";
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${String(Math.round(seconds - minutes * 60)).padStart(2, "0")}s`;
}

export function tokens(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : Math.round(value).toLocaleString("en-US");
}

export function money(value: number | null | undefined, currency = "USD"): string {
  if (value === null || value === undefined) return "—";
  const digits = Math.abs(value) >= 1 ? 2 : 4;
  return new Intl.NumberFormat("en-US", { style: "currency", currency, minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}

/** Signed difference for an operational value (time, tokens, cost). Neutral: no good/bad judgement. */
export function signedChange(delta: number | null, unit: "ms" | "tokens" | "currency", currency = "USD"): string {
  if (delta === null) return "—";
  if (delta === 0) return "no change";
  const sign = delta > 0 ? "+" : "−";
  const magnitude = Math.abs(delta);
  const body = unit === "ms" ? duration(magnitude) : unit === "tokens" ? tokens(magnitude) : money(magnitude, currency);
  return `${sign}${body}`;
}
