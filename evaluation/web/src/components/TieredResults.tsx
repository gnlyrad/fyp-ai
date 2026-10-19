import type { ReactNode } from "react";
import type { Row } from "../lib/api";
import { TIER_ORDER, splitTiers, type TierId } from "../lib/tiers";
import { useMetaApi } from "../lib/meta";
import { ResultList } from "./ResultList";
import { Disclosure, SectionTitle, cx } from "./ui";

export function useTierLabels(): Record<TierId, string> {
  const { presentation } = useMetaApi();
  return {
    missed: "Missed breaches",
    unclear: presentation.statuses.compliance_outcome_status.abstention.label,
    failure: "Prediction failures",
    other: "Other exceptions",
    success: "Successful results",
  };
}

const TIER_HELP: Partial<Record<TierId, string>> = {
  missed: "Actual breaches the system marked PASS.",
  other: "False alarms, checks marked N/A, and extraction checks where the model missed an expected turn.",
};

/** One tier of results. Exceptions are always open; successful results are collapsed
    when there is anything more important above them. */
export function TierSection({ tier, rows, label, show, flash = false, collapsible = false }: {
  tier: TierId;
  rows: Row[];
  label: string;
  show?: "call" | "question" | "both";
  flash?: boolean;
  /** Collapse the list (used for successes when exceptions exist). */
  collapsible?: boolean;
}) {
  const labels = useTierLabels();
  if (!rows.length) return null;
  const heading = `${labels[tier]} (${rows.length})`;
  const list = <ResultList rows={rows} show={show} label={`${labels[tier]} · ${label}`} />;
  return (
    <section className={cx("rounded-md", flash && "flash")} data-tier={tier}>
      {collapsible ? (
        <Disclosure title={<span className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">{heading}</span>}>
          {list}
        </Disclosure>
      ) : (
        <>
          <SectionTitle>{heading}</SectionTitle>
          {TIER_HELP[tier] && <p className="mb-2 -mt-1 text-xs text-ink-3">{TIER_HELP[tier]}</p>}
          {list}
        </>
      )}
    </section>
  );
}

/** A list of results, exceptions first, successes collapsed beneath them. */
export function TieredResults({ rows, label, show, empty = "No checks in this group.", header }: {
  rows: Row[];
  label: string;
  show?: "call" | "question" | "both";
  empty?: string;
  header?: ReactNode;
}) {
  if (!rows.length) return <p className="px-3 py-6 text-center text-sm text-ink-3">{empty}</p>;
  const tiers = splitTiers(rows);
  const hasExceptions = TIER_ORDER.some((tier) => tier !== "success" && tiers[tier].length > 0);
  return (
    <div className="space-y-5">
      {header}
      {TIER_ORDER.map((tier) => (
        <TierSection key={tier} tier={tier} rows={tiers[tier]} label={label} show={show}
          collapsible={tier === "success" && hasExceptions} />
      ))}
    </div>
  );
}
