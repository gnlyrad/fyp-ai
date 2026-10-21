import type { SttItem, SttUnit } from "./stt";

/* Presentation only: groups the exported display units into readable "change blocks" (Gold phrase / Transcribe phrase).
   Nothing is re-scored: a block's S/D/I are the sums of its units' exported parts, so the call's totals are unchanged. */

export type BlockLabel = "missing" | "extra" | "change";

export type Block = {
  id: number;
  turn_id: number;
  units: SttUnit[];
  gold: string;
  hyp: string;
  label: BlockLabel;
  substitutions: number;
  deletions: number;
  insertions: number;
  entity_errors: number;
  start: number | null;
  end: number | null;
  approximate: boolean;
};

export type Segment = { kind: "plain"; unit: SttUnit } | { kind: "block"; block: Block };

/** The distinct Transcribe items behind some units, in time order: an item shared by several reference words appears once. */
export function distinctItems(units: SttUnit[]): SttItem[] {
  const seen = new Map<number, SttItem>();
  for (const unit of units) for (const item of unit.items) if (!seen.has(item.item)) seen.set(item.item, item);
  return [...seen.values()].sort((a, b) => a.item - b.item);
}

export function hypText(units: SttUnit[]): string {
  return distinctItems(units).map((item) => item.content).join(" ");
}

function makeBlock(id: number, units: SttUnit[]): Block {
  const sum = (key: "substitutions" | "deletions" | "insertions" | "entity_errors") => units.reduce((total, unit) => total + unit[key], 0);
  const changed = units.filter((unit) => unit.errors > 0);
  const label: BlockLabel = changed.every((unit) => unit.op === "delete") ? "missing" : changed.every((unit) => unit.op === "insert" && !unit.ref_text) ? "extra" : "change";
  const starts = units.map((unit) => unit.start).filter((value): value is number => value !== null);
  const ends = units.map((unit) => unit.end).filter((value): value is number => value !== null);
  return {
    id, turn_id: units[0].turn_id, units,
    gold: units.map((unit) => unit.ref_text).filter(Boolean).join(" "),
    hyp: hypText(units),
    label,
    substitutions: sum("substitutions"), deletions: sum("deletions"), insertions: sum("insertions"), entity_errors: sum("entity_errors"),
    start: starts.length ? Math.min(...starts) : null, end: ends.length ? Math.max(...ends) : null,
    approximate: units.some((unit) => unit.approximate),
  };
}

/** Per turn, the sequence of plain (matching) units and change blocks. A block takes consecutive changed units of one turn and may
    bridge ONE matching unit between two changes, so a reordered phrase ("4 August" / "August 4th") reads as one comparison. */
export function buildSegments(units: SttUnit[]): { byTurn: Map<number, Segment[]>; blocks: Block[] } {
  const byTurn = new Map<number, Segment[]>();
  const blocks: Block[] = [];
  const turns = new Map<number, SttUnit[]>();
  for (const unit of units) turns.set(unit.turn_id, [...(turns.get(unit.turn_id) ?? []), unit]);
  for (const turnUnits of [...turns.entries()].sort((a, b) => a[0] - b[0]).map(([, list]) => list)) {
    const segments: Segment[] = [];
    let index = 0;
    while (index < turnUnits.length) {
      const unit = turnUnits[index];
      if (unit.errors === 0) {
        segments.push({ kind: "plain", unit });
        index += 1;
        continue;
      }
      const members = [unit];
      let next = index + 1;
      while (next < turnUnits.length) {
        if (turnUnits[next].errors > 0) {
          members.push(turnUnits[next]);
          next += 1;
        } else if (next + 1 < turnUnits.length && turnUnits[next + 1].errors > 0) {
          members.push(turnUnits[next], turnUnits[next + 1]);
          next += 2;
        } else {
          break;
        }
      }
      const block = makeBlock(blocks.length, members);
      blocks.push(block);
      segments.push({ kind: "block", block });
      index = next;
    }
    byTurn.set(turnUnits[0].turn_id, segments);
  }
  return { byTurn, blocks };
}

const wordCount = (text: string) => (text.trim() ? text.trim().split(/\s+/).length : 0);

/** A short change ("the → a", one missed or added word) reads inline; a longer one is stacked Gold above Transcribe. */
export function isCompact(block: Block): boolean {
  return wordCount(block.gold) <= 3 && wordCount(block.hyp) <= 3 && block.gold.length + block.hyp.length <= 32;
}

/** One word of the "now playing" strip: a timed Transcribe item, or a Gold word that was not transcribed (approximate time). */
export type TimelineWord = { key: string; text: string; start: number; end: number; unitIds: number[]; missing: boolean; approximate: boolean };

/** Every Transcribe item once, in time order, with missed Gold words placed where the export puts them. */
export function timeline(units: SttUnit[]): TimelineWord[] {
  const words = new Map<string, TimelineWord>();
  for (const unit of units) {
    for (const item of unit.items) {
      if (item.start === null) continue;
      const key = `i${item.item}`;
      const word = words.get(key);
      if (word) word.unitIds.push(unit.id);
      else words.set(key, { key, text: item.content, start: item.start, end: item.end ?? item.start, unitIds: [unit.id], missing: false, approximate: false });
    }
    if (!unit.items.length && unit.ref_text && unit.start !== null) {
      words.set(`u${unit.id}`, { key: `u${unit.id}`, text: unit.ref_text, start: unit.start, end: unit.end ?? unit.start, unitIds: [unit.id], missing: true, approximate: unit.approximate });
    }
  }
  return [...words.values()].sort((a, b) => a.start - b.start || Number(a.missing) - Number(b.missing));
}

export function breakdown(counts: { substitutions: number; deletions: number; insertions: number }): string {
  const parts: string[] = [];
  const name = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;
  if (counts.substitutions) parts.push(name(counts.substitutions, "substitution"));
  if (counts.deletions) parts.push(name(counts.deletions, "deletion"));
  if (counts.insertions) parts.push(name(counts.insertions, "insertion"));
  return parts.join(" · ") || "no scored errors";
}

/** True when a scoring-only rule made two DIFFERENT raw spellings count as equal ("9 February 1988." vs "February 9 1988"). When both sides
    are written identically ("I've" / "I've") the rule fired but changed nothing a person would notice, so it is not shown. */
export function ruleMatched(unit: SttUnit): boolean {
  const written = (text: string) => text.toLowerCase().replace(/[’‘]/g, "'").replace(/[.,!?;:"“”\s]/g, "");  // keeps $ € - ' and digits: those differences are the point
  return (unit.scoring_rules?.length ?? 0) > 0 && written(unit.ref_text) !== written(unit.hyp_text);
}

const CURRENCY_WORD: Record<string, string> = { dollars: "$", euros: "€", pounds: "£", rupees: "₹" };
const letters = (text: string) => text.toLowerCase().replace(/[^a-z0-9]/g, "");

/** Plain explanations of how a unit was compared, derived only from its exported parts and raw text. */
export function explain(unit: SttUnit): string[] {
  const notes: string[] = [];
  const equal = unit.parts.filter((part) => part.op === "equal");
  const inserted = unit.parts.filter((part) => part.op === "insert");
  if (equal.length && equal.every((part) => /^\d$|^point$/.test(part.ref ?? "")) && letters(unit.ref_text) !== letters(unit.hyp_text)) {
    notes.push("Scoring only: the digits match after normalisation.");
  } else if (unit.errors === 0 && unit.ref_text && letters(unit.ref_text) !== letters(unit.hyp_text)) {
    notes.push(`Scoring only: “${unit.ref_text}” and “${unit.hyp_text}” are written differently but match after normalisation, so this is not an error.`);
  }
  if (unit.ref_text && equal.length && inserted.length) {
    for (const part of inserted) {
      const symbol = CURRENCY_WORD[part.hyp ?? ""];
      notes.push(symbol && unit.hyp_text.includes(symbol)
        ? `The “${symbol}” in Transcribe's “${unit.hyp_text}” is read as “${part.hyp}”, which is not in Gold, so it counts as 1 insertion.`
        : `After normalisation Transcribe has an extra “${part.hyp}” here, counted as 1 insertion.`);
    }
  }
  return notes;
}
