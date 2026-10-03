import { pct } from "../lib/format";
import { cx } from "./ui";

export type Outcome = "correct" | "unclear" | "failure" | "wrong" | "neutral";

const FILL: Record<Outcome, string> = {
  correct: "bg-o-correct",
  unclear: "bg-o-unclear",
  failure: "bg-o-failure",
  wrong: "bg-o-wrong",
  neutral: "bg-o-neutral",
};

export type Segment = { key: string; label: string; count: number; outcome: Outcome; onClick?: () => void };

/** Stacked 100% bar with a 2px surface gap between segments and an always-present legend.
    Zero-count outcomes stay in the legend: "0 falsely cleared" is information. */
export function OutcomeBar({ segments, total, label }: { segments: Segment[]; total: number; label: string }) {
  const visible = segments.filter((s) => s.count > 0);
  return (
    <div>
      <div className="flex h-3 w-full gap-[2px] overflow-hidden rounded-[4px] bg-surface-2" role="img"
        aria-label={`${label}: ${segments.map((s) => `${s.count} ${s.label}`).join(", ")}`}>
        {visible.map((segment) => (
          <button key={segment.key} type="button" onClick={segment.onClick} disabled={!segment.onClick}
            title={`${segment.label}: ${segment.count} of ${total} (${pct((segment.count / total) * 100)})`}
            style={{ flexGrow: segment.count, flexBasis: 0 }}
            className={cx("focus-ring h-full min-w-[3px] transition-opacity hover:opacity-80", FILL[segment.outcome],
              segment.onClick ? "cursor-pointer" : "cursor-default")} />
        ))}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {segments.map((segment) => (
          <li key={segment.key}>
            <button type="button" onClick={segment.onClick} disabled={!segment.onClick || segment.count === 0}
              className={cx("inline-flex items-center gap-1.5",
                segment.onClick && segment.count > 0 ? "drill" : "cursor-default",
                segment.count === 0 && "opacity-60")}>
              <span className={cx("h-2.5 w-2.5 rounded-[3px]", FILL[segment.outcome])} />
              <span className="text-ink-2">{segment.label}</span>
              <span className="num font-semibold text-ink">{segment.count}</span>
              {total > 0 && <span className="num text-ink-3">({pct((segment.count / total) * 100)})</span>}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
