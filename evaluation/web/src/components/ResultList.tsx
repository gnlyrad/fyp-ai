import { ChevronRight } from "lucide-react";
import { Link, useLocation, useParams } from "react-router-dom";
import type { Row } from "../lib/api";
import { paths, useSource, type ResultNavState } from "../lib/url";
import { Badge, Empty } from "./ui";

export function verdictLine(row: Row): string {
  if (row.family === "compliance") {
    const predicted = row.predicted_verdict ?? (row.execution_status === "prediction_failure" ? "no result" : "—");
    return `Expected ${row.gold_verdict ?? "—"} · System ${predicted}`;
  }
  return row.expected_value === null ? "No value expected" : `Expected “${row.expected_value}”`;
}

/** Atomic call × question results. Every item opens the Individual Result, carrying this
    list so prev/next and Back follow it. */
export function ResultList({ rows, label, show = "both", empty = "No checks in this group." }: {
  rows: Row[];
  label: string;
  show?: "call" | "question" | "both";
  empty?: string;
}) {
  const source = useSource();
  const { datasetId = "", runId = "" } = useParams();
  const location = useLocation();
  if (!rows.length) return <Empty>{empty}</Empty>;
  const state: ResultNavState = {
    from: location.pathname + location.search,
    list: rows.map((row) => [row.call_id, row.question_id]),
    listLabel: label,
  };
  return (
    <ul className="divide-y divide-line rounded-md border border-line">
      {rows.map((row) => (
        <li key={`${row.call_id}/${row.question_id}`}>
          <Link to={paths.result(source, datasetId, runId, row.call_id, row.question_id)} state={state}
            className="focus-ring group flex items-center gap-3 px-3 py-2 hover:bg-surface-2">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 text-[13px]">
                <span className="font-medium text-ink">
                  {show === "question" ? row.question_id : show === "call" ? row.call_id : `${row.call_id} · ${row.question_id}`}
                </span>
                <span className="truncate text-ink-3">{verdictLine(row)}</span>
              </div>
              {show !== "call" && row.question && <div className="truncate text-xs text-ink-2">{row.question}</div>}
            </div>
            <Badge tone={row.outcome.tone}>{row.outcome.label}</Badge>
            <ChevronRight size={14} className="shrink-0 text-ink-3 opacity-50 group-hover:opacity-100" />
          </Link>
        </li>
      ))}
    </ul>
  );
}
