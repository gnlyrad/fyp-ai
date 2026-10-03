import { Check, X } from "lucide-react";
import type { SummaryNode } from "../lib/api";
import { useMetaApi } from "../lib/meta";
import { cx } from "./ui";

const LAYOUT: [string, string][] = [["TN", "FN"], ["FP", "TP"]];

/** 2x2 of completed PASS/FAIL decisions. Columns = ground truth, rows = system decision.
    Correct cells read blue with a check, errors red with a cross: never color alone. */
export function ConfusionMatrix({ node, onCell }: { node: SummaryNode; onCell: (category: string) => void }) {
  const { presentation, drillCount } = useMetaApi();
  return (
    <div className="grid grid-cols-[auto_1fr_1fr] gap-[2px] text-xs">
      <div />
      {["PASS", "FAIL"].map((gold) => (
        <div key={gold} className="px-2 pb-1 text-center font-medium text-ink-3">
          Actually {gold === "PASS" ? "compliant (PASS)" : "a breach (FAIL)"}
        </div>
      ))}
      {LAYOUT.map((row, index) => (
        <div key={index} className="contents">
          <div className="flex items-center justify-end pr-2 text-right font-medium text-ink-3">
            System said<br />{index === 0 ? "PASS" : "FAIL"}
          </div>
          {row.map((category) => {
            const correct = category === "TP" || category === "TN";
            const value = drillCount(node, `confusion.${category}`);
            const Icon = correct ? Check : X;
            return (
              <button key={category} type="button" onClick={() => onCell(category)} disabled={value === 0}
                title={`${presentation.confusion_categories[category].description} Click to list these checks.`}
                className={cx("focus-ring flex min-h-[64px] flex-col items-start justify-between rounded-md p-2.5 text-left transition",
                  correct ? "bg-tint-correct" : "bg-tint-wrong",
                  value > 0 ? "cursor-pointer hover:brightness-95" : "cursor-default opacity-70")}>
                <span className={cx("inline-flex items-center gap-1 font-medium", correct ? "text-accent-ink" : "text-ink-wrong")}>
                  <Icon size={13} strokeWidth={2.5} /> {presentation.confusion_categories[category].label}
                </span>
                <span className="num text-xl font-semibold text-ink">{value}</span>
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}
