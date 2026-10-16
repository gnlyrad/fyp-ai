import type { Row } from "./api";

/** How urgently a result needs a human, so every list can put the useful rows first. */
export type TierId = "missed" | "unclear" | "failure" | "other" | "success";

/** Display order everywhere: exceptions first, successes last. */
export const TIER_ORDER: TierId[] = ["missed", "unclear", "failure", "other", "success"];

export const EXCEPTION_TIERS: TierId[] = ["missed", "unclear", "failure", "other"];

/** Each row lands in exactly one tier, decided only from fields the evaluator already produced. */
export function tierOf(row: Row): TierId {
  if (row.execution_status === "prediction_failure") return "failure";
  if (row.confusion_category === "FN" || row.breach_outcome === "false_clearance") return "missed";
  if (row.compliance_outcome_status === "abstention") return "unclear";
  if (
    row.confusion_category === "FP" ||
    row.compliance_outcome_status === "unexpected_not_applicable" ||
    row.extraction_turn_status === "MISSED"
  ) {
    return "other";
  }
  return "success";
}

export function splitTiers(rows: Row[]): Record<TierId, Row[]> {
  const tiers: Record<TierId, Row[]> = { missed: [], unclear: [], failure: [], other: [], success: [] };
  for (const row of rows) tiers[tierOf(row)].push(row);
  return tiers;
}

/** The same result can belong to several metric populations; list it once. */
export function uniqueRows(rows: Row[]): Row[] {
  const seen = new Set<string>();
  return rows.filter((row) => {
    const key = `${row.call_id}/${row.question_id}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
