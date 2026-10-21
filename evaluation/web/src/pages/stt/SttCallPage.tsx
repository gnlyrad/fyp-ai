import { ChevronLeft, ChevronRight, Flag, Play, Square, Volume2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { Drawer } from "../../components/Drawer";
import { Page } from "../../components/Page";
import { Badge, Card, Disclosure, ErrorPanel, Loading, SectionTitle, Tip, cx } from "../../components/ui";
import { plural, sgt } from "../../lib/format";
import { sttPaths, useSttCall, useSttRun, type SttCall, type SttRun, type SttUnit } from "../../lib/stt";
import { breakdown, buildSegments, distinctItems, explain, hypText, isCompact, ruleMatched, timeline, type Block, type Segment, type TimelineWord } from "../../lib/sttBlocks";
import { useQueryState } from "../../lib/url";
import { SttKpi, clock, fill, groupLabel, plainOrDash, rate, sdi, useSttWords } from "./parts";

const ROLE: Record<string, string> = { EMPLOYEE: "Employee", CLIENT: "Client" };
const role = (value: string | null | undefined) => (value ? ROLE[value] ?? value : "—");
const OVERLAYS = ["speaker", "conf", "entity", "rules"] as const;
const DEFAULT_OVERLAYS: Overlay[] = ["speaker", "conf"];  // on until the person switches them off
type Overlay = (typeof OVERLAYS)[number];
const INSPECTOR_WIDTH = "lg:mr-[440px]";

/* ------------------------------------------------------------------ layout helpers */

/** Publishes an element's height as a CSS variable, so sticky offsets and jump targets follow wrapping. */
function useMeasuredHeight(element: HTMLElement | null, variable: string) {
  useEffect(() => {
    if (!element || typeof ResizeObserver === "undefined") return;
    const publish = () => document.documentElement.style.setProperty(variable, `${element.getBoundingClientRect().height}px`);
    publish();
    const observer = new ResizeObserver(publish);
    observer.observe(element);
    return () => observer.disconnect();
  }, [element, variable]);
}

function cssPixels(variable: string, fallback: number): number {
  const value = parseFloat(getComputedStyle(document.documentElement).getPropertyValue(variable));
  return Number.isFinite(value) ? value : fallback;
}

const JUMP_MARGIN = "calc(var(--stt-header-h, 48px) + var(--stt-toolbar-h, 56px) + 12px)";

/* ------------------------------------------------------------------ audio */

/** The playback position as a tiny external store: only the "now playing" strips re-render on every frame, not the transcript. */
type PlayClock = { get: () => number; set: (value: number) => void; subscribe: (listener: () => void) => () => void };
function createPlayClock(): PlayClock {
  let time = 0;
  const listeners = new Set<() => void>();
  return {
    get: () => time,
    set: (value) => { if (value !== time) { time = value; listeners.forEach((listener) => listener()); } },
    subscribe: (listener) => { listeners.add(listener); return () => { listeners.delete(listener); }; },
  };
}

type PlayRange = { start: number; end: number; from: number; to: number };

type Player = {
  available: boolean;
  failure: string | null;
  status: string | null;
  playing: boolean;
  range: PlayRange | null;
  clock: PlayClock;
  playRange: (start: number, end: number, pad?: number) => void;
  stop: () => void;
};

/** Playback state comes only from the media element's own events, never from an HTTP response. */
function useAudio(src: string | null, words: ReturnType<typeof useSttWords>) {
  const ref = useRef<HTMLAudioElement>(null);
  const stopAt = useRef<number | null>(null);
  const frame = useRef<number | null>(null);
  const playClock = useRef<PlayClock>(createPlayClock()).current;
  const [failure, setFailure] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [range, setRange] = useState<PlayRange | null>(null);

  const check = useCallback(() => {
    const audio = ref.current;
    if (!audio) return;
    playClock.set(audio.currentTime);
    if (stopAt.current !== null && audio.currentTime >= stopAt.current) {
      stopAt.current = null;
      audio.pause();
      return;
    }
    if (!audio.paused) frame.current = requestAnimationFrame(check);
  }, [playClock]);

  const startFrames = useCallback(() => {
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(check);
  }, [check]);

  const playRange = useCallback(async (start: number, end: number, pad = 2) => {
    const audio = ref.current;
    if (!audio) return;
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    try {
      if (audio.readyState < HTMLMediaElement.HAVE_METADATA) {
        await new Promise<void>((resolve, reject) => {
          const done = () => { clear(); resolve(); };
          const fail = () => { clear(); reject(new Error("the recording did not load")); };
          const clear = () => { audio.removeEventListener("loadedmetadata", done); audio.removeEventListener("error", fail); };
          audio.addEventListener("loadedmetadata", done);
          audio.addEventListener("error", fail);
          if (audio.networkState === HTMLMediaElement.NETWORK_EMPTY || audio.networkState === HTMLMediaElement.NETWORK_IDLE) audio.load();
        });
      }
      const from = Math.max(0, start - pad);
      const to = Math.min(end + pad, Number.isFinite(audio.duration) ? audio.duration : end + pad);
      audio.currentTime = from;
      playClock.set(from);
      stopAt.current = to;
      setRange({ start, end, from, to });
      setStatus(pad > 0 ? fill(words.audio?.playing_range, { from: clock(from), to: clock(to) }) : `Playing ${clock(from)}–${clock(to)}`);
      await audio.play();
      startFrames();
    } catch (error) {
      stopAt.current = null;
      setFailure(`${words.audio?.load_failed ?? "The recording could not be played"}: ${(error as Error).message}`);
    }
  }, [startFrames, playClock, words]);

  const stop = useCallback(() => {
    stopAt.current = null;
    ref.current?.pause();
  }, []);

  const onError = useCallback(async () => {
    if (!src) return;
    try {
      const response = await fetch(src, { headers: { Range: "bytes=0-0" } });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        setFailure(`${response.status}: ${body?.detail ?? response.statusText}`);
      } else {
        setFailure(`${words.audio?.load_failed ?? "The recording could not be played"}: the server sent it but the browser could not decode it.`);
      }
    } catch (error) {
      setFailure(`The audio endpoint could not be reached: ${(error as Error).message}`);
    }
  }, [src, words]);

  const handlers = {
    onPlay: () => {
      setPlaying(true);
      if (stopAt.current === null) { setRange(null); setStatus("Playing"); }  // the player's own button: follow along the whole call
      startFrames();
    },
    onPause: () => {
      setPlaying(false);
      stopAt.current = null;
      const audio = ref.current;
      if (audio) playClock.set(audio.currentTime);
      setStatus(audio ? `Paused at ${clock(audio.currentTime)}` : null);
    },
    onSeeked: () => { if (ref.current) playClock.set(ref.current.currentTime); },
    onTimeUpdate: check,
    onEnded: () => { setPlaying(false); setStatus("Finished"); },
    onError: () => { void onError(); },
  };
  return { ref, failure, status, playing, range, clock: playClock, playRange, stop, handlers };
}

/** Zoom-style "now playing" strip: Transcribe's timed words, the one under the playhead coloured, played words dark, the rest faint. */
function NowPlaying({ words, clock: playClock, from, to, follow, selected }: {
  words: TimelineWord[]; clock: PlayClock; from?: number; to?: number; follow?: boolean; selected?: Set<number>;
}) {
  const wording = useSttWords();
  const time = useSyncExternalStore(playClock.subscribe, playClock.get);
  let shown: TimelineWord[];
  if (follow) {  // the whole call is playing: a window that moves with the playhead
    let at = words.findIndex((word) => word.start > time);
    at = at === -1 ? words.length : at;
    shown = words.slice(Math.max(0, at - 14), at + 16);
  } else {
    shown = words.filter((word) => word.end > (from ?? 0) - 0.01 && word.start < (to ?? Infinity) + 0.01);
  }
  const current = shown.find((word) => !word.missing && word.start <= time && time < word.end + 0.04)?.key;
  if (!shown.length) return null;
  return (
    <div className="mt-2 rounded-md border border-line bg-surface-2 px-3 py-2" data-testid="now-playing">
      <div className="mb-1 flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wide text-ink-3">
        {wording.audio?.now_playing ?? "Now playing"}
        <Tip content={{ title: wording.audio?.now_playing ?? "Now playing", what: wording.audio?.now_playing_help }} />
        <span className="num ml-auto font-normal normal-case tracking-normal">{clock(time)}</span>
      </div>
      <p className="text-[14px] leading-7">
        {shown.map((word) => {
          const isCurrent = word.key === current;
          const played = word.end <= time;
          const inSelection = selected && word.unitIds.some((id) => selected.has(id));
          return (
            <span key={word.key}>
              <span data-current={isCurrent || undefined}
                className={cx("rounded px-0.5 transition-colors duration-75",
                  isCurrent ? "bg-accent text-white" : word.missing ? "text-o-failure line-through" : played ? "text-ink" : "text-ink-3",
                  inSelection && !isCurrent && "underline decoration-2 underline-offset-4 decoration-accent")}>
                {word.text}
              </span>{" "}
            </span>
          );
        })}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ transcript pieces */

function wordStyle(unit: SttUnit, overlays: Set<Overlay>) {
  return cx(
    overlays.has("entity") && unit.entity_groups.length > 0 && "border-b-2 border-accent",
    overlays.has("speaker") && unit.speaker_mismatch && "underline decoration-dotted decoration-2 decoration-[color:var(--o-both)] underline-offset-4",
    overlays.has("conf") && unit.low_confidence && !unit.speaker_mismatch && "underline decoration-dotted decoration-2 decoration-ink-3 underline-offset-4",
    overlays.has("rules") && ruleMatched(unit) && !(overlays.has("conf") && unit.low_confidence) && !(overlays.has("speaker") && unit.speaker_mismatch)
      && "underline decoration-dashed decoration-1 decoration-accent underline-offset-4",
  );
}

function PlainWord({ unit, selected, overlays, onSelect }: { unit: SttUnit; selected: boolean; overlays: Set<Overlay>; onSelect: () => void }) {
  return (
    <button type="button" id={`unit-${unit.id}`} onClick={onSelect} style={{ scrollMarginTop: JUMP_MARGIN }}
      className={cx("focus-ring rounded-sm px-[1px] text-left align-baseline hover:bg-surface-2", wordStyle(unit, overlays), selected && "bg-tint-correct ring-2 ring-accent")}>
      {unit.ref_text}
    </button>
  );
}

const HYP_TONE: Record<Block["label"], string> = { change: "text-ink-wrong", missing: "text-o-failure", extra: "text-ink-warn" };
const CHIP_TONE: Record<Block["label"], string> = {
  change: "bg-tint-wrong",
  missing: "border border-dashed border-o-failure",
  extra: "bg-tint-warn",
};

/** The Gold → Transcribe reading of a block, without labels: "the → a", a struck-through missed word, or "+added". */
function Comparison({ block, speaker }: { block: Block; speaker?: boolean }) {
  const labels = useSttWords().diff?.labels ?? {};
  const hypClass = cx("font-semibold", HYP_TONE[block.label], speaker && "underline decoration-dotted decoration-2 decoration-[color:var(--o-both)] underline-offset-4");
  if (block.label === "missing") return <span className="text-o-failure line-through decoration-2">{block.gold}</span>;
  if (block.label === "extra") return <span className={hypClass}>+{block.hyp}</span>;
  return (
    <>
      <span className="text-ink-2">{block.gold || <i className="text-ink-3">{labels.extra_gold}</i>}</span>
      <span className="mx-1 text-ink-3" aria-label="became">→</span>
      <span className={hypClass}>{block.hyp || <i className="font-normal text-ink-3">{labels.missing_hyp}</i>}</span>
    </>
  );
}

function ChangeBlock({ block, selected, overlays, onSelect }: { block: Block; selected: boolean; overlays: Set<Overlay>; onSelect: () => void }) {
  const labels = useSttWords().diff?.labels ?? {};
  const entity = overlays.has("entity") && block.units.some((unit) => unit.entity_groups.length > 0);
  const speaker = overlays.has("speaker") && block.units.some((unit) => unit.speaker_mismatch);
  const common = {
    type: "button" as const, id: `block-${block.id}`, onClick: onSelect, style: { scrollMarginTop: JUMP_MARGIN },
    title: `${labels[block.label] ?? block.label}: ${breakdown(block)}. Click for details.`, "data-block": block.id,
  };
  if (isCompact(block)) {
    return (
      <button {...common} className={cx("focus-ring rounded px-1 py-px text-left align-baseline [box-decoration-break:clone] hover:brightness-95",
        CHIP_TONE[block.label], entity && "border-b-2 border-b-accent", selected && "ring-2 ring-accent")}>
        <Comparison block={block} speaker={speaker} />
      </button>
    );
  }
  return (  // a longer change: Gold above Transcribe, no labels (the reading order is explained once above the transcript)
    <button {...common} className={cx("focus-ring mx-0.5 my-0.5 inline-flex max-w-full flex-col rounded-md border-l-[3px] px-2 py-0.5 text-left align-middle text-[13px] leading-5 hover:brightness-95",
      block.label === "missing" ? "border-l-o-failure bg-surface-2" : block.label === "extra" ? "border-l-o-unclear bg-tint-warn" : "border-l-o-wrong bg-tint-wrong",
      entity && "border-b-2 border-b-accent", selected && "ring-2 ring-accent")}>
      <span className={cx("min-w-0 break-words", block.label === "missing" ? "text-o-failure line-through" : "text-ink-2")}>
        {block.gold || <i className="text-ink-3">{labels.extra_gold}</i>}
      </span>
      <span className={cx("min-w-0 break-words font-semibold", HYP_TONE[block.label],
        speaker && "underline decoration-dotted decoration-2 decoration-[color:var(--o-both)] underline-offset-4")}>
        {block.hyp || <i className="font-normal text-ink-3">{labels.missing_hyp}</i>}
      </span>
    </button>
  );
}

function TurnHeader({ turn, mismatches, overlays }: { turn: SttCall["turns"][number]; mismatches: number; overlays: Set<Overlay> }) {
  return (
    <div className="pt-1 text-xs leading-4">
      <div className="font-semibold text-ink-2">{role(turn.speaker)}</div>
      <div className="num text-ink-3">Turn {turn.turn_id} · {clock(turn.start)}</div>
      {overlays.has("speaker") && mismatches > 0 && (
        <div className="mt-1 inline-flex items-center gap-1 rounded bg-surface-2 px-1 text-[10px] font-medium text-ink-2" title="Words given to the wrong speaker">
          <Flag size={10} className="text-[color:var(--o-both)]" />{plural(mismatches, "word")}
        </div>
      )}
    </div>
  );
}

const countMismatches = (segments: Segment[]) => segments.reduce((n, segment) => n + (segment.kind === "plain" ? Number(segment.unit.speaker_mismatch)
  : segment.block.units.filter((unit) => unit.speaker_mismatch).length), 0);

function InlineTranscript({ call, byTurn, filter, overlays, selectedUnit, selectedBlock, select }: {
  call: SttCall; byTurn: Map<number, Segment[]>; filter: string; overlays: Set<Overlay>;
  selectedUnit: number | null; selectedBlock: number | null; select: (unitId: number) => void;
}) {
  return (
    <div className="space-y-3">
      {call.turns.map((turn) => {
        const segments = byTurn.get(turn.turn_id) ?? [];
        const blockAt = segments.map((segment) => segment.kind === "block");
        if (filter === "errors" && !blockAt.some(Boolean)) return null;
        const near = (index: number) => blockAt.some((isBlock, k) => isBlock && Math.abs(k - index) <= 3);
        return (
          <div key={turn.turn_id} className="grid grid-cols-[76px_minmax(0,1fr)] gap-3">
            <TurnHeader turn={turn} mismatches={countMismatches(segments)} overlays={overlays} />
            <p className="min-w-0 break-words text-[14px] leading-7 text-ink">
              {segments.map((segment, index) => {
                if (filter === "errors" && !near(index)) {
                  return index > 0 && near(index - 1) ? <span key={`gap-${index}`} className="text-ink-3"> … </span> : null;
                }
                return (
                  <span key={segment.kind === "block" ? `b${segment.block.id}` : `u${segment.unit.id}`}>
                    {index > 0 && " "}
                    {segment.kind === "block"
                      ? <ChangeBlock block={segment.block} selected={selectedBlock === segment.block.id} overlays={overlays} onSelect={() => select(segment.block.units[0].id)} />
                      : <PlainWord unit={segment.unit} selected={selectedUnit === segment.unit.id} overlays={overlays} onSelect={() => select(segment.unit.id)} />}
                  </span>
                );
              })}
            </p>
          </div>
        );
      })}
    </div>
  );
}

/** One side of a side-by-side row: plain text with each change highlighted on that side only. */
function SideText({ segments, side, selectedBlock, select }: {
  segments: Segment[]; side: "gold" | "hyp"; selectedBlock: number | null; select: (unitId: number) => void;
}) {
  const labels = useSttWords().diff?.labels ?? {};
  const seen = new Set<number>();  // a Transcribe item shared by several Gold words is written once
  const hypOf = (units: SttUnit[]) => {
    const fresh = distinctItems(units).filter((item) => !seen.has(item.item));
    fresh.forEach((item) => seen.add(item.item));
    return fresh.map((item) => item.content).join(" ");
  };
  const pieces: ReactNode[] = [];
  for (const segment of segments) {
    if (segment.kind === "plain") {
      const text = side === "gold" ? segment.unit.ref_text : hypOf([segment.unit]);
      if (text) pieces.push(<span key={`u${segment.unit.id}`}>{text}</span>);
      continue;
    }
    const block = segment.block;
    const text = side === "gold" ? block.gold : hypOf(block.units);
    const absent = !text;
    if (absent && (side === "gold" ? block.label === "extra" : block.label === "missing")) {
      pieces.push(  // nothing on this side: a small caret marks where the other side has words
        <button key={`b${block.id}`} type="button" onClick={() => select(block.units[0].id)} data-side-block={block.id}
          title={side === "gold" ? labels.extra_gold : labels.missing_hyp}
          className={cx("focus-ring rounded px-0.5 font-semibold", block.label === "extra" ? "text-ink-warn" : "text-o-failure", selectedBlock === block.id && "ring-2 ring-accent")}>‸</button>,
      );
      continue;
    }
    pieces.push(
      <button key={`b${block.id}`} type="button" onClick={() => select(block.units[0].id)} data-side-block={block.id} title={breakdown(block)}
        className={cx("focus-ring rounded px-0.5 text-left [box-decoration-break:clone]",
          side === "gold"
            ? block.label === "missing" ? "text-o-failure line-through decoration-2" : "bg-tint-wrong"
            : cx("font-semibold", block.label === "extra" ? "bg-tint-warn text-ink-warn" : "bg-tint-wrong text-ink-wrong"),
          selectedBlock === block.id && "ring-2 ring-accent")}>
        {text}
      </button>,
    );
  }
  return <>{pieces.flatMap((piece, index) => (index ? [" ", piece] : [piece]))}</>;
}

/** The turn being heard right now (-1 when nothing plays), as an external-store snapshot so only a change of turn re-renders. */
function useHeardTurn(player: Player, turns: SttCall["turns"]): number {
  const turnAt = (time: number) => turns.find((turn) => turn.start !== null && turn.end !== null && time >= turn.start && time < turn.end)?.turn_id ?? -1;
  return useSyncExternalStore(player.clock.subscribe, () => (player.playing ? turnAt(player.clock.get()) : -1));
}

function SideBySide({ call, byTurn, filter, selectedBlock, selectedTurn, select, player }: {
  call: SttCall; byTurn: Map<number, Segment[]>; filter: string; selectedBlock: number | null; selectedTurn: number | null; select: (unitId: number) => void;
  player: Player;
}) {
  const rows = useSttWords().diff?.rows ?? {};
  const heard = useHeardTurn(player, call.turns);
  const columns = "md:grid-cols-[88px_minmax(0,1fr)_minmax(0,1fr)]";
  return (
    <div className="text-[14px] leading-7">
      <div className={cx("hidden gap-x-5 border-b border-line pb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-3 md:grid", columns)}>
        <span>Turn</span>
        <span className="flex items-center gap-1">{rows.gold ?? "Gold"}<Tip content={{ title: rows.gold ?? "Gold", what: rows.gold_help }} /></span>
        <span className="flex items-center gap-1">{rows.hyp ?? "Transcribe"}<Tip content={{ title: rows.hyp ?? "Transcribe", what: rows.hyp_help }} /></span>
      </div>
      {call.turns.map((turn) => {
        const segments = byTurn.get(turn.turn_id) ?? [];
        const changes = segments.filter((segment) => segment.kind === "block").length;
        if (filter === "errors" && !changes) return null;
        const units = segments.flatMap((segment) => (segment.kind === "block" ? segment.block.units : [segment.unit]));
        const labels = [...new Set(distinctItems(units).map((item) => `${item.speaker ?? "?"} → ${role(item.mapped_role)}`))];
        return (
          <div key={turn.turn_id} id={`turn-${turn.turn_id}`} style={{ scrollMarginTop: JUMP_MARGIN }}
            className={cx("grid gap-x-5 gap-y-1 border-b border-line py-2 last:border-b-0", columns,
              heard === turn.turn_id ? "bg-tint-correct" : selectedTurn === turn.turn_id && "bg-surface-2")}
            data-heard={heard === turn.turn_id || undefined}>
            <div className="text-xs leading-5">
              <div className="font-semibold text-ink-2">Turn {turn.turn_id}</div>
              <div className="text-ink-3">{role(turn.speaker)}</div>
              <div className="num text-ink-3">{clock(turn.start)}</div>
              {player.available && turn.start !== null && turn.end !== null && (
                heard === turn.turn_id
                  ? <button type="button" onClick={player.stop} aria-label={`Stop turn ${turn.turn_id}`}
                      className="focus-ring mt-0.5 inline-flex items-center gap-1 rounded border border-line px-1.5 py-0.5 text-ink-2 hover:text-ink"><Square size={10} />Stop</button>
                  : <button type="button" onClick={() => player.playRange(turn.start as number, turn.end as number, 0)} aria-label={`Play turn ${turn.turn_id}`}
                      className="focus-ring mt-0.5 inline-flex items-center gap-1 rounded bg-accent px-1.5 py-0.5 font-medium text-white"><Play size={10} />Play</button>
              )}
              {changes > 0 && <div className="mt-0.5"><Badge tone="bad" icon={false}>{plural(changes, "change")}</Badge></div>}
            </div>
            <div className="min-w-0 break-words text-ink">
              <span className="mr-2 text-[10px] font-semibold uppercase tracking-wide text-ink-3 md:hidden">{rows.gold ?? "Gold"}</span>
              <SideText segments={segments} side="gold" selectedBlock={selectedBlock} select={select} />
            </div>
            <div className="min-w-0 break-words text-ink">
              <span className="mr-2 text-[10px] font-semibold uppercase tracking-wide text-ink-3 md:hidden">{rows.hyp ?? "Transcribe"}</span>
              <SideText segments={segments} side="hyp" selectedBlock={selectedBlock} select={select} />
              {labels.length > 0 && <div className="font-mono text-[11px] leading-4 text-ink-3">{labels.join(", ")}</div>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ drawer */

const PART_LABEL = { equal: "Matches", substitute: "Substituted", delete: "Missing", insert: "Extra" } as const;
const PART_TONE = { equal: "text-accent-ink", substitute: "text-ink-wrong", delete: "text-o-failure", insert: "text-ink-warn" } as const;

function AlignmentTable({ units }: { units: SttUnit[] }) {
  const rows = useSttWords().diff?.rows ?? {};
  return (
    <div className="space-y-2">
      {units.map((unit) => (
        <div key={unit.id} className="rounded-md border border-line">
          <div className="border-b border-line bg-surface-2 px-2 py-1 text-xs text-ink-2">
            {rows.gold ?? "Gold"} “{unit.ref_text || "∅"}” · {rows.hyp ?? "Transcribe"} “{unit.hyp_text || "∅"}”
            {unit.errors > 0 && <span className="ml-1 text-ink-wrong">· {breakdown(unit)}</span>}
          </div>
          <table className="w-full text-[12px]">
            <tbody>
              {unit.parts.map((part, index) => (
                <tr key={index} className="border-t border-line first:border-t-0">
                  <td className="w-1/3 px-2 py-0.5 font-mono">{part.ref ?? "∅"}</td>
                  <td className="w-1/3 px-2 py-0.5 font-mono">{part.hyp ?? "∅"}</td>
                  <td className={cx("px-2 py-0.5 text-right font-medium", PART_TONE[part.op])}>
                    {PART_LABEL[part.op]}{part.entity && <span className="ml-1 font-normal text-ink-3">(entity WER)</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}

function Listen({ start, end, approximate, player, label, words, selected }: {
  start: number | null; end: number | null; approximate: boolean; player: Player; label: string; words: TimelineWord[]; selected: Set<number>;
}) {
  const wording = useSttWords();
  const mine = player.range && start !== null && player.range.start === start && player.range.end === (end ?? start);
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2 text-[13px]">
        <button type="button" disabled={!player.available || start === null} onClick={() => start !== null && player.playRange(start, end ?? start)}
          className="focus-ring inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 font-medium text-white disabled:opacity-40">
          <Play size={14} /> {label}
        </button>
        <button type="button" disabled={!player.available} onClick={player.stop}
          className="focus-ring inline-flex items-center gap-1 rounded-md border border-line px-2.5 py-1.5 text-ink-2 enabled:hover:text-ink disabled:opacity-40">
          <Square size={12} /> Stop
        </button>
        <span className="num text-ink-2">{clock(start)} – {clock(end)}</span>
        {approximate && <Tip content={{ title: "Approximate position", what: wording.diff?.approximate_time }}><Badge tone="neutral" icon={false}>approximate</Badge></Tip>}
      </div>
      {player.available && player.status && <p className="mt-1 text-xs text-ink-2">{player.status}</p>}
      {!player.available && <p className="mt-1 text-xs text-ink-wrong">{player.failure}</p>}
      {mine && player.range && <NowPlaying words={words} clock={player.clock} from={player.range.from} to={player.range.to} selected={selected} />}
    </div>
  );
}

function SelectionDrawer({ call, run, units, block, player, blocks, words: timelineWords, onClose, goto }: {
  call: SttCall; run: SttRun | undefined; units: SttUnit[]; block: Block | null; player: Player; blocks: Block[]; words: TimelineWord[];
  onClose: () => void; goto: (unitId: number) => void;
}) {
  const words = useSttWords();
  const { datasetId = "" } = useParams();
  const rows = words.diff?.rows ?? {};
  const labels = words.diff?.labels ?? {};
  const items = distinctItems(units);
  const notes = units.flatMap(explain);
  const firstError = units.flatMap((unit) => unit.parts).find((part) => part.op !== "equal");
  const group = firstError && run?.errors.find((entry) => entry.op === firstError.op && entry.ref === firstError.ref && entry.hyp === firstError.hyp);
  const others = group?.occurrences.filter((o) => !(o.call_id === call.call_id && units.some((unit) => unit.id === o.unit))) ?? [];
  const previous = block ? blocks[block.id - 1] : [...blocks].reverse().find((b) => b.units[0].id < units[0].id);
  const next = block ? blocks[block.id + 1] : blocks.find((b) => b.units[0].id > units[0].id);
  const refRoles = [...new Set(units.filter((unit) => unit.ref_role).map((unit) => role(unit.ref_role)))];
  const entityErrors = units.reduce((n, unit) => n + unit.entity_errors, 0);
  const appliedRules = [...new Set(units.filter(ruleMatched).flatMap((unit) => unit.scoring_rules ?? []))];
  const start = block ? block.start : units[0].start;
  return (
    <Drawer open inspector onClose={onClose}
      title={block ? fill(words.diff?.navigation, { index: block.id + 1, total: blocks.length }) : "Matched word"}
      subtitle={`${call.call_id} · turn ${units[0].turn_id} · ${clock(start)}`}>
      <div className="flex gap-2 text-xs">
        <button type="button" disabled={!previous} onClick={() => previous && goto(previous.units[0].id)}
          className="focus-ring inline-flex items-center gap-1 rounded border border-line px-2 py-1 text-ink-2 enabled:hover:text-ink disabled:opacity-40"><ChevronLeft size={13} />Previous</button>
        <button type="button" disabled={!next} onClick={() => next && goto(next.units[0].id)}
          className="focus-ring inline-flex items-center gap-1 rounded border border-line px-2 py-1 text-ink-2 enabled:hover:text-ink disabled:opacity-40">Next<ChevronRight size={13} /></button>
        <span className="self-center text-ink-3">keys: j / k</span>
      </div>

      <div className="rounded-md border border-line p-3" data-testid="comparison">
        <dl className="space-y-1 text-[14px]">
          <div className="grid grid-cols-[84px_minmax(0,1fr)] gap-2">
            <dt className="flex items-center gap-1 text-xs text-ink-3">{rows.gold ?? "Gold"}<Tip content={{ title: rows.gold ?? "Gold", what: rows.gold_help }} /></dt>
            <dd className={cx("break-words text-ink", block?.label === "missing" && "text-o-failure line-through")}>
              {(block ? block.gold : units[0].ref_text) || <span className="italic text-ink-3">{labels.extra_gold}</span>}</dd>
          </div>
          <div className="grid grid-cols-[84px_minmax(0,1fr)] gap-2">
            <dt className="flex items-center gap-1 text-xs text-ink-3">{rows.hyp ?? "Transcribe"}<Tip content={{ title: rows.hyp ?? "Transcribe", what: rows.hyp_help }} /></dt>
            <dd className={cx("break-words font-semibold", block ? HYP_TONE[block.label] : "text-ink")}>
              {hypText(units) || <span className="font-normal italic text-ink-3">{labels.missing_hyp}</span>}</dd>
          </div>
        </dl>
        <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-line pt-2 text-xs">
          {block ? <Badge tone="bad" icon={false}>{breakdown(block)}</Badge> : <Badge tone="good" icon={false}>matches Gold</Badge>}
          {entityErrors > 0 && <span className="text-ink-2">{entityErrors} counted in entity WER</span>}
          {units.some((unit) => unit.entity_groups.length) && [...new Set(units.flatMap((unit) => unit.entity_groups))].map((g) => (
            <Badge key={g} tone="neutral" icon={false}>{groupLabel(g)}</Badge>
          ))}
        </div>
        {notes.length > 0 && <ul className="mt-2 space-y-0.5 text-xs text-ink-2">{notes.map((note) => <li key={note}>{note}</li>)}</ul>}
        {appliedRules.length > 0 && (
          <div className="mt-2 border-t border-line pt-2 text-xs text-ink-2" data-testid="scoring-rules">
            <p className="mb-1 font-medium text-ink">{words.normalisation?.scoring_only}</p>
            <ul className="space-y-0.5">
              {appliedRules.map((id) => (
                <li key={id}><span className="font-medium text-ink">{words.normalisation?.rules[id]?.label ?? id}.</span> {words.normalisation?.rules[id]?.help}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <Listen start={start} end={block ? block.end : units[0].end} approximate={block ? block.approximate : units[0].approximate}
        player={player} label={block ? "Play this phrase" : "Play this word"} words={timelineWords} selected={new Set(units.map((unit) => unit.id))} />

      <Disclosure title="Exact alignment and normalisation">
        <AlignmentTable units={units} />
        <p className="mt-2 text-xs text-ink-3">{words.diff?.highlight_note}</p>
      </Disclosure>

      <div>
        <SectionTitle>Speaker and confidence</SectionTitle>
        <p className="mb-1 text-[13px] text-ink-2">
          Gold speaker: <span className="font-medium text-ink">{refRoles.join(", ") || "— (not in Gold)"}</span>
          {units.some((unit) => unit.speaker_mismatch) && <span className="ml-2"><Badge tone="warn" icon={false}>given to the wrong speaker</Badge></span>}
        </p>
        {items.length > 0 ? (
          <table className="w-full text-[13px]">
            <thead><tr className="text-left text-[11px] text-ink-3"><th className="px-2 font-medium">{rows.hyp ?? "Transcribe"} word</th><th className="px-2 font-medium">Speaker label</th><th className="px-2 text-right font-medium">Confidence</th></tr></thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.item} className="even:bg-surface-2">
                  <td className="px-2 py-0.5 font-mono text-xs">{item.content}</td>
                  <td className="px-2 py-0.5 text-xs text-ink-2"><span className="font-mono">{item.speaker ?? "?"}</span> → {item.mapped_role ? role(item.mapped_role) : "no role"}</td>
                  <td className={cx("num px-2 py-0.5 text-right", item.confidence !== null && item.confidence < 0.5 && "font-medium text-ink-warn")}>{item.confidence ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : <p className="text-xs text-ink-3">No {rows.hyp ?? "Transcribe"} words here.</p>}
      </div>

      {others.length > 0 && group && (
        <div>
          <SectionTitle>Same error elsewhere in this run ({others.length})</SectionTitle>
          <ul className="space-y-1 text-[13px]">
            {others.slice(0, 12).map((o) => (
              <li key={`${o.call_id}-${o.unit}`}>
                <Link to={sttPaths.call(datasetId, call.run_id, o.call_id, o.unit)} className="focus-ring block rounded px-2 py-1 hover:bg-surface-2">
                  <span className="font-medium text-ink-2">{o.call_id}</span> <span className="text-ink-3">…{o.context.before}</span>{" "}
                  <span className="font-medium text-ink-wrong">{group.op === "insert" ? `+${group.hyp}` : group.op === "delete" ? <s>{group.ref}</s> : `${plainOrDash(group.ref)} → ${plainOrDash(group.hyp)}`}</span>
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}

      <Disclosure title={`Raw ${rows.hyp ?? "Transcribe"} output for the whole call`}>
        <p className="whitespace-pre-wrap text-xs leading-5 text-ink-2">{call.raw_hypothesis_text}</p>
      </Disclosure>
    </Drawer>
  );
}

/* ------------------------------------------------------------------ toolbar */

function Toggle({ pressed, onClick, children, title }: { pressed: boolean; onClick: () => void; children: ReactNode; title?: string }) {
  return (
    <button type="button" aria-pressed={pressed} onClick={onClick} title={title}
      className={cx("focus-ring rounded px-2.5 py-1 font-medium", pressed ? "bg-accent text-white" : "text-ink-2 hover:text-ink")}>{children}</button>
  );
}

function Legend() {
  const words = useSttWords();
  const legend = words.diff?.legend ?? {};
  const overlays = words.diff?.overlays ?? {};
  return (
    <div className="grid gap-x-6 gap-y-1.5 border-t border-line pt-2 text-xs text-ink-2 sm:grid-cols-2">
      <div className="flex items-start gap-2"><span className="shrink-0 rounded bg-tint-wrong px-1">the <span className="text-ink-3">→</span> <b className="text-ink-wrong">a</b></span>{legend.change}</div>
      <div className="flex items-start gap-2"><span className="shrink-0 rounded border border-dashed border-o-failure px-1 text-o-failure line-through">word</span>{legend.missing}</div>
      <div className="flex items-start gap-2"><span className="shrink-0 rounded bg-tint-warn px-1 font-semibold text-ink-warn">+word</span>{legend.extra}</div>
      <div className="flex items-start gap-2"><span className="shrink-0 px-1.5">word</span>{legend.plain}</div>
      <div className="flex items-start gap-2"><span className="shrink-0 border-b-2 border-accent px-1">entity</span>{overlays.entity}</div>
      <div className="flex items-start gap-2"><span className="shrink-0 px-1 underline decoration-dotted decoration-2 decoration-[color:var(--o-both)] underline-offset-4">speaker</span>{overlays.speaker}</div>
      <div className="flex items-start gap-2"><span className="shrink-0 px-1 underline decoration-dotted decoration-2 decoration-ink-3 underline-offset-4">unsure</span>{overlays.low_confidence}</div>
      <div className="flex items-start gap-2"><span className="shrink-0 px-1 underline decoration-dashed decoration-1 decoration-accent underline-offset-4">rule</span>{overlays.rules}</div>
    </div>
  );
}

/* ------------------------------------------------------------------ page */

export default function SttCallPage() {
  const { datasetId = "", runId = "", callId = "" } = useParams();
  const words = useSttWords();
  const query = useSttCall(runId, callId);
  const runQuery = useSttRun(runId);
  const [params, setParams] = useQueryState();
  const filter = params.get("filter") ?? "all";
  const view = params.get("view") === "side" || params.get("view") === "turns" ? "side" : "inline";
  const highlightParam = params.get("hl");  // absent = the defaults; "none" = everything off
  const overlays = useMemo(() => new Set<Overlay>(highlightParam === null ? DEFAULT_OVERLAYS
    : highlightParam.split(",").filter((x): x is Overlay => (OVERLAYS as readonly string[]).includes(x))), [highlightParam]);
  const [legendOpen, setLegendOpen] = useState(false);
  const unitParam = params.get("unit");
  const selectedUnit = unitParam !== null && unitParam !== "" ? Number(unitParam) : null;
  const call = query.data;
  const [toolbar, setToolbar] = useState<HTMLDivElement | null>(null);
  const [optionsOpen, setOptionsOpen] = useState(false);
  useMeasuredHeight(toolbar, "--stt-toolbar-h");

  const audioAvailable = Boolean(call?.audio_status.available);
  const audioSrc = audioAvailable ? sttPaths.audio(runId, callId) : null;
  const audio = useAudio(audioSrc, words);
  const player: Player = {
    available: audioAvailable && !audio.failure,
    failure: audio.failure ?? call?.audio_status.reason ?? null,
    status: audio.status, playing: audio.playing, range: audio.range, clock: audio.clock,
    playRange: (s, e, pad) => { void audio.playRange(s, e, pad); }, stop: audio.stop,
  };

  const { byTurn, blocks } = useMemo(() => (call ? buildSegments(call.units) : { byTurn: new Map<number, Segment[]>(), blocks: [] as Block[] }), [call]);
  const timelineWords = useMemo(() => (call ? timeline(call.units) : []), [call]);
  const blockOfUnit = useMemo(() => {
    const map = new Map<number, Block>();
    for (const block of blocks) for (const unit of block.units) map.set(unit.id, block);
    return map;
  }, [blocks]);
  const selectedBlock = selectedUnit !== null ? blockOfUnit.get(selectedUnit) ?? null : null;
  const select = useCallback((unitId: number | null, replace = false) => setParams({ unit: unitId === null ? null : String(unitId) }, { replace }), [setParams]);
  const toggleOverlay = (overlay: Overlay) => {
    const next = new Set(overlays);
    if (next.has(overlay)) next.delete(overlay); else next.add(overlay);
    setParams({ hl: [...next].join(",") || "none" }, { replace: true });
  };

  useEffect(() => {  // j / k step through change blocks
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || (event.target as HTMLElement | null)?.closest("input, textarea, select, audio")) return;
      if (event.key !== "j" && event.key !== "k") return;
      const current = selectedBlock ? selectedBlock.id : -1;
      const target = event.key === "j"
        ? blocks.find((b) => (selectedBlock ? b.id > current : selectedUnit === null || b.units[0].id > selectedUnit))
        : [...blocks].reverse().find((b) => (selectedBlock ? b.id < current : selectedUnit !== null && b.units[0].id < selectedUnit));
      if (target) setParams({ unit: String(target.units[0].id) }, { replace: true });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [blocks, selectedBlock, selectedUnit, setParams]);

  const selectedTurn = selectedUnit !== null ? call?.units.find((unit) => unit.id === selectedUnit)?.turn_id ?? null : null;
  useEffect(() => {  // a jump lands below the header and the sticky toolbar (scrollIntoView avoided: it misbehaved with drawers here)
    if (selectedUnit === null || !call) return;
    const id = view === "side" ? `turn-${selectedTurn}` : selectedBlock ? `block-${selectedBlock.id}` : `unit-${selectedUnit}`;
    const element = document.getElementById(id);
    if (!element) return;
    const offset = cssPixels("--stt-header-h", 48) + cssPixels("--stt-toolbar-h", 56) + 12;
    const top = element.getBoundingClientRect().top;
    if (view === "side" && top >= offset && element.getBoundingClientRect().bottom <= window.innerHeight) return;  // the row is already in view
    window.scrollTo({ top: Math.max(0, top + window.scrollY - offset), behavior: "smooth" });
  }, [selectedUnit, selectedBlock, selectedTurn, call, view]);

  const crumbs = [
    { label: "Datasets", to: sttPaths.home() },
    { label: datasetId.replace(/_/g, " "), to: sttPaths.dataset(datasetId) },
    { label: runQuery.data ? sgt(runQuery.data.collected_at, { seconds: false }) : runId, to: sttPaths.run(datasetId, runId) },
    { label: callId },
  ];
  if (query.isPending) return <Page section="stt" crumbs={crumbs}><Loading label="Loading call" /></Page>;
  if (query.isError || !call) return <Page section="stt" crumbs={crumbs}><ErrorPanel error={query.error} /></Page>;

  const wer = call.metrics.wer;
  const entity = call.metrics.entity_wer;
  const meanConfidence = call.metrics.confidence.items ? Math.round((call.metrics.confidence.sum / call.metrics.confidence.items) * 10000) / 10000 : null;
  const selectedUnits = selectedBlock ? selectedBlock.units : selectedUnit !== null ? call.units.filter((unit) => unit.id === selectedUnit) : [];
  const inspecting = selectedUnits.length > 0;
  const position = selectedBlock ? selectedBlock.id + 1 : null;
  const previousBlock = selectedBlock ? blocks[selectedBlock.id - 1] : undefined;
  const nextBlock = selectedBlock ? blocks[selectedBlock.id + 1] : blocks.find((b) => selectedUnit === null || b.units[0].id > selectedUnit);
  const views = words.diff?.views ?? {};

  return (
    <Page section="stt" crumbs={crumbs}>
      <div className={cx(inspecting && INSPECTOR_WIDTH)}>
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <h1 className="text-xl font-semibold text-ink">{call.call_id}</h1>
          <span className="text-xs text-ink-3">{plural(blocks.length, "change")} · {plural(wer.errors, "scored error")}</span>
          <div className="ml-auto flex gap-2 text-xs">
            {call.previous_call && <Link className="focus-ring inline-flex items-center gap-1 rounded border border-line px-2 py-1 text-ink-2 hover:text-ink" to={sttPaths.call(datasetId, runId, call.previous_call)}><ChevronLeft size={13} />{call.previous_call}</Link>}
            {call.next_call && <Link className="focus-ring inline-flex items-center gap-1 rounded border border-line px-2 py-1 text-ink-2 hover:text-ink" to={sttPaths.call(datasetId, runId, call.next_call)}>{call.next_call}<ChevronRight size={13} /></Link>}
          </div>
        </div>

        <div className="mb-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          <SttKpi id="normalised_wer" value={rate(wer.wer)} caption={`${wer.errors} errors · ${sdi(wer)}`} />
          {entity ? <SttKpi id="entity_wer" value={rate(entity.wer)} caption={`${entity.errors} errors in ${entity.reference_words} words`} />
            : <SttKpi id="entity_wer" value="Not available" caption="no annotations for this dataset" />}
          <SttKpi id="average_confidence" value={meanConfidence === null ? "—" : String(meanConfidence)} />
          <SttKpi id="speaker_attribution" value={rate(call.metrics.speaker.accuracy)} caption={`${call.metrics.speaker.correct_items} of ${call.metrics.speaker.eligible_items} words`} />
        </div>

        <Card className="mb-3 p-3">
          <div className="flex flex-wrap items-center gap-3">
            <Volume2 size={16} className="text-ink-3" />
            {audioSrc ? (
              <audio ref={audio.ref} controls preload="metadata" src={audioSrc} className="h-8 max-w-full" {...audio.handlers} />
            ) : null}
            {player.available ? (
              <span className="text-xs text-ink-2">{audio.status ?? "Ready: press play, or play any selected word or phrase."}</span>
            ) : (
              <span className="min-w-0 text-[13px]">
                <span className="font-medium text-ink-wrong">{player.failure ?? words.audio?.unavailable}</span>
                {!call.audio_status.available && <span className="block text-xs text-ink-3">{words.audio?.unavailable_help}</span>}
              </span>
            )}
          </div>
          {player.available && audio.playing && !audio.range && <NowPlaying words={timelineWords} clock={audio.clock} follow />}
        </Card>

        <Card className="p-4">
          <div ref={setToolbar} className="sticky z-10 -mx-4 -mt-4 mb-3 rounded-t-lg border-b border-line bg-surface px-4 py-2" style={{ top: "var(--stt-header-h, 48px)" }}>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <div role="group" aria-label="Show" className="flex rounded-md border border-line p-0.5">
                <Toggle pressed={filter === "all"} onClick={() => setParams({ filter: null }, { replace: true })}>All words</Toggle>
                <Toggle pressed={filter === "errors"} onClick={() => setParams({ filter: "errors" }, { replace: true })}>Errors only</Toggle>
              </div>
              <button type="button" aria-expanded={optionsOpen} onClick={() => setOptionsOpen(!optionsOpen)}
                className="focus-ring rounded-md border border-line px-2.5 py-1 font-medium text-ink-2 hover:text-ink sm:hidden">{optionsOpen ? "Fewer options" : "View & highlights"}</button>
              <div className={cx("flex-wrap items-center gap-2", optionsOpen ? "flex" : "hidden sm:flex")}>
                <div role="group" aria-label="View" className="flex rounded-md border border-line p-0.5">
                  <Toggle pressed={view === "inline"} onClick={() => setParams({ view: null }, { replace: true })}>{views.inline ?? "Inline"}</Toggle>
                  <Toggle pressed={view === "side"} onClick={() => setParams({ view: "side" }, { replace: true })} title={views.side_help}>{views.side ?? "Side by side"}</Toggle>
                </div>
                {view === "inline" && (
                  <div role="group" aria-label="Highlight" className="flex flex-wrap rounded-md border border-line p-0.5">
                    <Toggle pressed={overlays.has("speaker")} onClick={() => toggleOverlay("speaker")} title={words.diff?.overlays?.speaker}>Speaker issues</Toggle>
                    <Toggle pressed={overlays.has("conf")} onClick={() => toggleOverlay("conf")} title={words.diff?.overlays?.low_confidence}>Low confidence</Toggle>
                    <Toggle pressed={overlays.has("entity")} onClick={() => toggleOverlay("entity")} title={words.diff?.overlays?.entity}>Entity spans</Toggle>
                    <Toggle pressed={overlays.has("rules")} onClick={() => toggleOverlay("rules")} title={words.diff?.overlays?.rules}>Normalised matches</Toggle>
                  </div>
                )}
              </div>
              <div className="flex items-center gap-1" aria-label="Change navigation">
                <button type="button" disabled={!previousBlock} onClick={() => previousBlock && select(previousBlock.units[0].id, true)} aria-label="Previous change"
                  className="focus-ring rounded border border-line p-1 text-ink-2 enabled:hover:text-ink disabled:opacity-40"><ChevronLeft size={14} /></button>
                <span className="num min-w-[104px] text-center font-medium text-ink" data-testid="change-counter">
                  {fill(words.diff?.navigation, { index: position ?? "–", total: blocks.length })}
                </span>
                <button type="button" disabled={!nextBlock} onClick={() => nextBlock && select(nextBlock.units[0].id, true)} aria-label="Next change"
                  className="focus-ring rounded border border-line p-1 text-ink-2 enabled:hover:text-ink disabled:opacity-40"><ChevronRight size={14} /></button>
              </div>
              <span className="hidden text-ink-3 xl:inline" data-testid="scored-totals">{fill(words.diff?.scored_totals, { substitutions: wer.substitutions, deletions: wer.deletions, insertions: wer.insertions })}</span>
              <span className="num text-ink-3 xl:hidden" title={fill(words.diff?.scored_totals, { substitutions: wer.substitutions, deletions: wer.deletions, insertions: wer.insertions })}>Scored: {sdi(wer)}</span>
              <button type="button" aria-expanded={legendOpen} onClick={() => setLegendOpen(!legendOpen)} className="focus-ring ml-auto rounded px-1 font-medium text-accent-ink hover:underline">
                {legendOpen ? "Hide legend" : "Legend"}
              </button>
            </div>
            {legendOpen && <div className="mt-2"><Legend /></div>}
          </div>

          <p className="mb-3 text-xs text-ink-3" data-testid="reading-key">{view === "side" ? views.side_help : words.diff?.reading_key}</p>
          {view === "side"
            ? <SideBySide call={call} byTurn={byTurn} filter={filter} selectedBlock={selectedBlock ? selectedBlock.id : null} selectedTurn={selectedTurn} select={(unitId) => select(unitId)} player={player} />
            : <InlineTranscript call={call} byTurn={byTurn} filter={filter} overlays={overlays} selectedUnit={selectedUnit}
                selectedBlock={selectedBlock ? selectedBlock.id : null} select={(unitId) => select(unitId)} />}
          <p className="mt-4 border-t border-line pt-2 text-xs text-ink-3">{words.diff?.highlight_note}</p>
        </Card>
      </div>

      {inspecting && (
        <SelectionDrawer call={call} run={runQuery.data} units={selectedUnits} block={selectedBlock} player={player} blocks={blocks} words={timelineWords}
          onClose={() => select(null, true)} goto={(unitId) => select(unitId, true)} />
      )}
    </Page>
  );
}
