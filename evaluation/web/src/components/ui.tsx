import * as Tooltip from "@radix-ui/react-tooltip";
import { AlertTriangle, CheckCircle2, ChevronDown, CircleSlash, HelpCircle, Info, Loader2, XCircle } from "lucide-react";
import { useState, type ReactNode } from "react";
import type { Ratio, Tone } from "../lib/api";
import { fraction, ratioPct } from "../lib/format";
import { useMetaApi } from "../lib/meta";

export function cx(...names: (string | false | null | undefined)[]) {
  return names.filter(Boolean).join(" ");
}

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return <section className={cx("rounded-lg border border-line bg-surface", className)}>{children}</section>;
}

export function SectionTitle({ children, aside, id }: { children: ReactNode; aside?: ReactNode; id?: string }) {
  return (
    <div className="mb-2 flex items-baseline justify-between gap-3" id={id}>
      <h3 className="text-[11px] font-semibold uppercase tracking-wider text-ink-3">{children}</h3>
      {aside && <div className="text-xs text-ink-3">{aside}</div>}
    </div>
  );
}

/* ---------------------------------------------------------------- tooltips */

type TipContent = { title: string; what?: string; calculation?: string; important?: string; technical?: ReactNode };

export function Tip({ content, children }: { content: TipContent; children?: ReactNode }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>
        {children ?? (
          <button type="button" aria-label={`What is ${content.title}?`}
            className="focus-ring inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-ink-3 hover:text-ink">
            <Info size={13} strokeWidth={2} />
          </button>
        )}
      </Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content side="top" align="start" sideOffset={6} collisionPadding={12}
          className="z-50 max-w-[340px] rounded-lg border border-line bg-surface p-3 text-[13px] leading-snug text-ink-2 shadow-lg">
          <div className="mb-1 font-semibold text-ink">{content.title}</div>
          {content.what && <p>{content.what}</p>}
          {content.calculation && (
            <p className="mt-2"><span className="font-medium text-ink">How it's calculated: </span>{content.calculation}</p>
          )}
          {content.important && (
            <p className="mt-2"><span className="font-medium text-ink">Important: </span>{content.important}</p>
          )}
          {content.technical && <div className="mt-2 border-t border-line pt-2 text-xs text-ink-3">{content.technical}</div>}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

/** Plain-English metric tooltip from metric_presentation.json; the formal definition is secondary. */
export function MetricTip({ id }: { id: string }) {
  const { wording, definition } = useMetaApi();
  const words = wording(id);
  const def = definition(id);
  return (
    <Tip content={{
      title: words.label,
      what: words.what,
      calculation: words.calculation,
      important: words.important,
      technical: def && (
        <>
          <div><span className="font-medium">Formula:</span> <span className="font-mono">{def.formula}</span></div>
          <div className="mt-0.5"><span className="font-medium">Population:</span> {def.population}</div>
        </>
      ),
    }} />
  );
}

export function OperationLabel({ id }: { id: string }) {
  const { operationWording } = useMetaApi();
  const words = operationWording(id);
  return (
    <span className="inline-flex items-center gap-1">
      {words.label}
      <Tip content={{ title: words.label, what: words.what, calculation: words.calculation, important: words.important }} />
    </span>
  );
}

/** A metric's name with its explanation tip. `text` shortens the visible name (e.g. in a narrow column header). */
export function MetricLabel({ id, className, text }: { id: string; className?: string; text?: string }) {
  const { wording } = useMetaApi();
  return (
    <span className={cx("inline-flex items-center gap-1", className)}>
      {text ?? wording(id).label}
      <MetricTip id={id} />
    </span>
  );
}

/* ---------------------------------------------------------------- KPI tile */

export function Kpi({ id, ratio, onClick, size = "md", caption }: {
  id: string;
  ratio: Ratio | null;
  onClick?: () => void;
  size?: "md" | "sm";
  caption?: string;
}) {
  const value = ratioPct(ratio);
  const body = (
    <>
      <div className={cx("num font-semibold text-ink", size === "md" ? "text-[22px] leading-7" : "text-lg leading-6")}>{value}</div>
      <div className="num text-xs text-ink-3">{ratio ? caption ?? fraction(ratio) : "Not applicable"}</div>
    </>
  );
  return (
    <div className="min-w-0 rounded-md border border-line bg-surface px-3 py-2">
      <div className="mb-0.5 text-xs text-ink-2"><MetricLabel id={id} /></div>
      {onClick && ratio ? (
        <button type="button" onClick={onClick} className="drill block w-full" title="Show the checks behind this number">{body}</button>
      ) : body}
    </div>
  );
}

/* ---------------------------------------------------------------- badges */

const TONE_CLASS: Record<Tone, string> = {
  good: "bg-tint-correct text-accent-ink",
  bad: "bg-tint-wrong text-ink-wrong",
  warn: "bg-tint-warn text-ink-warn",
  neutral: "bg-surface-2 text-ink-2",
};

const TONE_ICON: Record<Tone, typeof CheckCircle2> = {
  good: CheckCircle2,
  bad: XCircle,
  warn: AlertTriangle,
  neutral: CircleSlash,
};

export function Badge({ tone, children, icon = true, title }: { tone: Tone; children: ReactNode; icon?: boolean; title?: string }) {
  const Icon = TONE_ICON[tone];
  return (
    <span title={title} className={cx("inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-xs font-medium", TONE_CLASS[tone])}>
      {icon && <Icon size={12} strokeWidth={2.25} />}
      {children}
    </span>
  );
}

/* ---------------------------------------------------------------- states */

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 p-6 text-sm text-ink-3">
      <Loader2 size={16} className="animate-spin" /> {label}…
    </div>
  );
}

export function ErrorPanel({ error, action }: { error: unknown; action?: ReactNode }) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <Card className="p-4">
      <div className="flex items-start gap-2 text-sm">
        <XCircle size={16} className="mt-0.5 shrink-0 text-o-wrong" />
        <div>
          <div className="font-medium text-ink">Could not load this view</div>
          <div className="mt-0.5 text-ink-2">{message}</div>
          {action && <div className="mt-3">{action}</div>}
        </div>
      </div>
    </Card>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="px-3 py-6 text-center text-sm text-ink-3">{children}</div>;
}

export function Disclosure({ title, children, defaultOpen = false, className }: {
  title: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
  className?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={className}>
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}
        className="focus-ring inline-flex items-center gap-1 rounded text-xs font-medium text-ink-2 hover:text-ink">
        <ChevronDown size={14} className={cx("transition-transform", !open && "-rotate-90")} />
        {title}
      </button>
      {open && <div className="mt-2">{children}</div>}
    </div>
  );
}

/** A readable extraction field name; the internal identifier stays available on hover. */
export function FieldName({ field, className }: { field: string | null | undefined; className?: string }) {
  const { fieldLabel } = useMetaApi();
  if (!field) return null;
  return <span className={className} title={`Internal field name: ${field}`}>{fieldLabel(field)}</span>;
}

export function Json({ value }: { value: unknown }) {
  return (
    <pre className="max-h-96 overflow-auto rounded-md border border-line bg-surface-2 p-3 font-mono text-xs text-ink-2">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export function Help({ children }: { children: ReactNode }) {
  return (
    <p className="flex items-start gap-1.5 text-xs text-ink-3">
      <HelpCircle size={13} className="mt-px shrink-0" />
      <span>{children}</span>
    </p>
  );
}
