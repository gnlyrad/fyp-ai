import { BarChart3, ChevronRight } from "lucide-react";
import type { ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";
import type { Source } from "../lib/api";
import { paths, useSource } from "../lib/url";
import { Tip, cx } from "./ui";

export type Crumb = { label: string; to?: string };

const SOURCE_HELP: Record<Source, string> = {
  shared: "Published benchmark runs stored in Supabase, the same for everyone.",
  local: "Runs in evaluation/runs on the machine serving this dashboard, including unpublished debug runs.",
};

export function Page({ crumbs = [], children }: { crumbs?: Crumb[]; children: ReactNode }) {
  const source = useSource();
  const navigate = useNavigate();
  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-20 border-b border-line bg-surface/95 backdrop-blur">
        <div className="mx-auto flex h-12 max-w-[1440px] items-center gap-3 px-4 sm:px-6">
          <Link to={paths.datasets(source)} className="focus-ring flex shrink-0 items-center gap-2 rounded font-semibold text-ink">
            <BarChart3 size={18} className="text-accent" />
            <span className="hidden sm:inline">Evaluation Dashboard</span>
          </Link>
          <nav aria-label="Breadcrumb" className="flex min-w-0 flex-1 items-center gap-1 text-[13px] text-ink-2">
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
        </div>
      </header>
      <main className="mx-auto max-w-[1440px] px-4 py-5 sm:px-6">{children}</main>
    </div>
  );
}
