import { createContext, useContext, useMemo, type ReactNode } from "react";
import type { Segment } from "../../components/OutcomeBar";
import type { ChecklistItem, Row, RunDetail, SummaryNode } from "../../lib/api";
import { useMetaApi } from "../../lib/meta";
import { useQueryState } from "../../lib/url";

/* Drill-down panels are addressed by URL:
     ?panel=call:CALL_002&focus=breaches     whole call, one dimension focused
     ?panel=question:C01&focus=decision      whole question, one dimension focused
     ?panel=metric:<metric id>&scope=...     the population behind one metric
     ?panel=drill:<drill key>&scope=...      one outcome group (e.g. confusion.FN)
     ?panel=error:<index>                    one diagnostic
   scope = call:<id> | question:<id>; omitted = whole run. */

export type Focus = "decision" | "breaches" | "abstentions" | "failures" | "extraction";
export type Scope = { kind: "call" | "question"; id: string } | null;

type RunView = {
  data: RunDetail;
  questions: Record<string, ChecklistItem>;
  callIds: string[];
  rowsFor: (scope: Scope) => Row[];
  nodeFor: (scope: Scope) => SummaryNode;
  scopeLabel: (scope: Scope) => string;
  open: (panel: string, extra?: { focus?: Focus; scope?: Scope }) => void;
  close: () => void;
};

const RunContext = createContext<RunView | null>(null);

export const scopeParam = (scope: Scope) => (scope ? `${scope.kind}:${scope.id}` : null);

export function parseScope(value: string | null): Scope {
  if (!value) return null;
  const [kind, ...rest] = value.split(":");
  return kind === "call" || kind === "question" ? { kind, id: rest.join(":") } : null;
}

export function RunProvider({ data, children }: { data: RunDetail; children: ReactNode }) {
  const [, setParams] = useQueryState();
  const value = useMemo<RunView>(() => {
    const questions = Object.fromEntries(data.checklist.map((item) => [item.id, item]));
    const callIds = Object.keys(data.summary.by_call_id ?? {}).sort();
    return {
      data,
      questions,
      callIds,
      rowsFor: (scope) => !scope ? data.rows
        : data.rows.filter((row) => (scope.kind === "call" ? row.call_id : row.question_id) === scope.id),
      nodeFor: (scope) => !scope ? data.summary.overall
        : (scope.kind === "call" ? data.summary.by_call_id : data.summary.by_question_id)[scope.id] ?? {},
      scopeLabel: (scope) => !scope ? "Whole run" : scope.kind === "call" ? scope.id : `${scope.id} across all calls`,
      open: (panel, extra = {}) => setParams({ panel, focus: extra.focus ?? null, scope: scopeParam(extra.scope ?? null) }),
      close: () => setParams({ panel: null, focus: null, scope: null }),
    };
  }, [data, setParams]);
  return <RunContext.Provider value={value}>{children}</RunContext.Provider>;
}

export function useRunView(): RunView {
  const value = useContext(RunContext);
  if (!value) throw new Error("RunProvider missing");
  return value;
}

/** Outcome-bar segments, built from the drill registry so every count is the summary's own. */
export function useSegments() {
  const { drillCount } = useMetaApi();
  const { open } = useRunView();
  const { drills } = useMetaApi();
  const segment = (node: SummaryNode, key: string, outcome: Segment["outcome"], scope: Scope): Segment => ({
    key,
    label: drills[key].label,
    count: drillCount(node, key),
    outcome,
    onClick: () => open(`drill:${key}`, { scope }),
  });
  return {
    decisions: (node: SummaryNode, scope: Scope): Segment[] => [
      segment(node, "compliance.binary_decision", "correct", scope),
      segment(node, "compliance.abstention", "unclear", scope),
      segment(node, "compliance.prediction_failure", "failure", scope),
      segment(node, "compliance.unexpected_not_applicable", "neutral", scope),
    ],
    breaches: (node: SummaryNode, scope: Scope): Segment[] => [
      segment(node, "breach.detected", "correct", scope),
      segment(node, "breach.abstained", "unclear", scope),
      segment(node, "breach.prediction_failure", "failure", scope),
      segment(node, "breach.unexpected_not_applicable", "neutral", scope),
      segment(node, "breach.false_clearance", "wrong", scope),
    ],
    extraction: (node: SummaryNode, scope: Scope): Segment[] => [
      segment(node, "extraction.found", "correct", scope),
      segment(node, "extraction.not_found", "wrong", scope),
      segment(node, "extraction.coverage_failure", "failure", scope),
    ],
  };
}

/** The outcome groups that partition each metric's population, for the metric panel. */
export const METRIC_POPULATIONS: Record<string, string[]> = {
  "compliance.decision_coverage": ["compliance.binary_decision", "compliance.abstention", "compliance.prediction_failure", "compliance.unexpected_not_applicable"],
  "compliance.abstention_rate": ["compliance.abstention", "compliance.binary_decision", "compliance.prediction_failure", "compliance.unexpected_not_applicable"],
  "compliance.prediction_failure_rate": ["compliance.prediction_failure", "compliance.binary_decision", "compliance.abstention", "compliance.unexpected_not_applicable"],
  "compliance.unexpected_not_applicable_rate": ["compliance.unexpected_not_applicable", "compliance.binary_decision", "compliance.abstention", "compliance.prediction_failure"],
  "compliance.binary_accuracy": ["confusion.TP", "confusion.TN", "confusion.FN", "confusion.FP"],
  "compliance.breach_recall": ["confusion.TP", "confusion.FN"],
  "compliance.breach_precision": ["confusion.TP", "confusion.FP"],
  "compliance.automatic_breach_detection_rate": ["breach.detected", "breach.false_clearance", "breach.abstained", "breach.prediction_failure", "breach.unexpected_not_applicable"],
  "compliance.false_clearance_rate": ["breach.false_clearance", "breach.detected", "breach.abstained", "breach.prediction_failure", "breach.unexpected_not_applicable"],
  "compliance.gold_breach_abstention_rate": ["breach.abstained", "breach.detected", "breach.false_clearance", "breach.prediction_failure", "breach.unexpected_not_applicable"],
  "compliance.gold_breach_prediction_failure_rate": ["breach.prediction_failure", "breach.detected", "breach.false_clearance", "breach.abstained", "breach.unexpected_not_applicable"],
  "extraction.evidence_coverage": ["extraction.found", "extraction.not_found", "extraction.coverage_failure"],
  "extraction.prediction_failure_rate": ["extraction.all"],
};
