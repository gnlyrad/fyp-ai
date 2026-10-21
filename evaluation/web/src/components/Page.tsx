import { BarChart3, ChevronRight, HardDrive } from "lucide-react";
import { useEffect, useRef, type ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";
import type { Source } from "../lib/api";
import { useMetaApi } from "../lib/meta";
import { paths, useSource } from "../lib/url";
import { Tip, cx } from "./ui";

export type Crumb = { label: string; to?: string };
export type Section = "nlp" | "stt";

const SOURCE_HELP: Record<Source, string> = {
  shared: "Published benchmark runs stored in Supabase, the same for everyone.",
  local: "Runs in evaluation/runs on the machine serving this dashboard, including unpublished debug runs.",
};

const NLP_SOURCE_KEY = "eval.nlpSource";

/** The source the NLP tab was last on, so switching to Speech-to-text and back keeps your place. Storage may be unavailable. */
function rememberedSource(fallback: Source): Source {
  try {
    const stored = window.localStorage.getItem(NLP_SOURCE_KEY);
    return stored === "shared" || stored === "local" ? stored : fallback;
  } catch {
    return fallback;
  }
}

export function Page({ crumbs = [], children, section = "nlp" }: { crumbs?: Crumb[]; children: ReactNode; section?: Section }) {
  const source = useSource();
  const navigate = useNavigate();
  const { default_source, stt_presentation } = useMetaApi();
  useEffect(() => {
    if (section !== "nlp") return;
    try {
      window.localStorage.setItem(NLP_SOURCE_KEY, source);
    } catch {
      /* private window or blocked storage: the tab just falls back to the default source */
    }
  }, [section, source]);

  // The header's real height (it wraps on narrow screens) is published as --stt-header-h for sticky elements below it.
  const headerRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const header = headerRef.current;
    if (!header || typeof ResizeObserver === "undefined") return;
    const publish = () => document.documentElement.style.setProperty("--stt-header-h", `${header.getBoundingClientRect().height}px`);
    publish();
    const observer = new ResizeObserver(publish);
    observer.observe(header);
    return () => observer.disconnect();
  }, []);

  const tabs: { id: Section; label: string; to: string }[] = [
    { id: "stt", label: stt_presentation.tab_label ?? "Speech-to-text", to: "/stt" },
    { id: "nlp", label: "NLP pipeline", to: paths.datasets(section === "nlp" ? source : rememberedSource(default_source)) },
  ];
  const home = section === "stt" ? "/stt" : paths.datasets(source);
  const localBadge = stt_presentation.local_badge;
  return (
    <div className="min-h-screen">
      <header ref={headerRef} className="sticky top-0 z-20 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex min-h-12 max-w-[1440px] flex-wrap items-center gap-x-3 gap-y-1 px-4 py-1.5 sm:px-6">
          <Link to={home} className="focus-ring flex shrink-0 items-center gap-2 rounded font-semibold text-ink">
            <BarChart3 size={18} className="text-accent" />
            <span className="hidden sm:inline">Evaluation Dashboard</span>
          </Link>
          <nav aria-label="Section" className="flex shrink-0 gap-1 text-[13px]">
            {tabs.map((tab) => (
              <Link key={tab.id} to={tab.to} aria-current={section === tab.id ? "page" : undefined}
                className={cx("focus-ring rounded-md px-2.5 py-1 font-medium",
                  section === tab.id ? "bg-accent text-white" : "text-ink-2 hover:bg-surface-2 hover:text-ink")}>
                {tab.label}
              </Link>
            ))}
          </nav>
          <nav aria-label="Breadcrumb" className="flex min-w-0 flex-1 basis-48 items-center gap-1 text-[13px] text-ink-2">
            {crumbs.map((crumb, index) => (
              <span key={index} className="flex min-w-0 items-center gap-1">
                <ChevronRight size={14} className="shrink-0 text-ink-3" />
                {crumb.to ? (
                  <Link to={crumb.to} className="focus-ring truncate rounded hover:text-ink hover:underline">{crumb.label}</Link>
                ) : (
                  <span className="truncate font-medium text-ink">{crumb.label}</span>
                )}
              </span>
            ))}
          </nav>
          {section === "stt" ? (
            <Tip content={{ title: localBadge?.label ?? "Local results", what: localBadge?.help }}>
              <span tabIndex={0} className="focus-ring inline-flex shrink-0 items-center gap-1 rounded-md border border-line px-2.5 py-1 text-xs font-medium text-ink-2">
                <HardDrive size={13} /> {localBadge?.label ?? "Local results"}
              </span>
            </Tip>
          ) : (
            <div role="group" aria-label="Source" className="flex shrink-0 rounded-md border border-line p-0.5 text-xs">
              {(["shared", "local"] as Source[]).map((option) => (
                <Tip key={option} content={{ title: option === "shared" ? "Shared" : "Local", what: SOURCE_HELP[option] }}>
                  <button type="button" aria-pressed={source === option}
                    onClick={() => option !== source && navigate(paths.datasets(option))}
                    className={cx("focus-ring rounded px-2.5 py-1 font-medium",
                      source === option ? "bg-accent text-white" : "text-ink-2 hover:text-ink")}>
                    {option === "shared" ? "Shared" : "Local"}
                  </button>
                </Tip>
              ))}
            </div>
          )}
        </div>
      </header>
      <main className="mx-auto max-w-[1440px] px-4 py-5 sm:px-6">{children}</main>
    </div>
  );
}
