import { createContext, useContext, useMemo, type ReactNode } from "react";
import type { Meta, MetricDefinition, MetricWording, Ratio, Row, SummaryNode } from "./api";
import { at, count, matches } from "./format";

type MetaApi = Meta & {
  definition: (id: string) => MetricDefinition;
  wording: (id: string) => MetricWording;
  operationWording: (id: string) => MetricWording;
  /** Human name for an internal extraction field (client_identification_number -> Client NRIC / FIN). */
  fieldLabel: (field: string | null | undefined) => string;
  /** A metric's ratio read from any aggregate node via its summary_path (never recomputed). */
  metric: (node: SummaryNode | null | undefined, id: string) => Ratio | null;
  drillCount: (node: SummaryNode | null | undefined, key: string) => number;
  drillRows: (rows: Row[], key: string) => Row[];
  status: (group: string, id: string | null | undefined) => string;
};

const MetaContext = createContext<MetaApi | null>(null);

export function MetaProvider({ meta, children }: { meta: Meta; children: ReactNode }) {
  const value = useMemo<MetaApi>(() => {
    const index = Object.fromEntries(meta.definitions.metrics.map((m) => [m.id, m]));
    return {
      ...meta,
      definition: (id) => index[id],
      wording: (id) => meta.presentation.metrics[id] ?? { label: index[id]?.label ?? id, audience: "developer", what: "", calculation: "" },
      operationWording: (id) => meta.presentation.operations[id],
      fieldLabel: (field) => {
        if (!field) return "";
        const known = meta.presentation.fields?.[field]?.label;
        if (known) return known;
        const spaced = field.replace(/_/g, " ").trim();  // unknown future field: readable fallback
        return spaced.charAt(0).toUpperCase() + spaced.slice(1);
      },
      metric: (node, id) => {
        const value = at(node, index[id].summary_path);
        return value && typeof value === "object" && "denominator" in value ? (value as Ratio) : null;
      },
      drillCount: (node, key) => {
        const path = meta.drills[key].count_path;  // none for outcomes derived from turn ids: count their rows
        return path ? count(node, path) : 0;
      },
      drillRows: (rows, key) => rows.filter((row) => matches(row, meta.drills[key].filter)),
      status: (group, id) => (id ? meta.presentation.statuses[group]?.[id]?.label ?? id : "—"),
    };
  }, [meta]);
  return <MetaContext.Provider value={value}>{children}</MetaContext.Provider>;
}

export function useMetaApi(): MetaApi {
  const value = useContext(MetaContext);
  if (!value) throw new Error("MetaProvider missing");
  return value;
}
