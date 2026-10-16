import { EyeOff } from "lucide-react";
import { useUnlinkedRuns } from "../lib/api";
import { plural, sgt, shortHash } from "../lib/format";
import { useSource } from "../lib/url";
import { Card, Disclosure } from "./ui";

/** Local runs that exist on disk but cannot be shown under any dataset, with the reason.
    Without this they would disappear silently. Renders nothing when every run is listed. */
export function HiddenRunsNotice() {
  const source = useSource();
  const hidden = useUnlinkedRuns(source);
  if (!hidden.data?.length) return null;
  const missing = hidden.data.filter((run) => run.reason === "dataset_missing");
  const unreadable = hidden.data.filter((run) => run.reason !== "dataset_missing");
  return (
    <Card className="mb-4 p-3">
      <div className="flex items-start gap-2 text-[13px]">
        <EyeOff size={15} className="mt-0.5 shrink-0 text-ink-3" />
        <div className="min-w-0">
          <div className="font-medium text-ink">{plural(hidden.data.length, "local run")} not listed</div>
          {missing.length > 0 && (
            <p className="mt-0.5 text-ink-2">
              {plural(missing.length, "run was", "runs were")} made on a different version of the dataset that is no longer in{" "}
              <span className="font-mono text-xs">evaluation/datasets</span>, so {missing.length === 1 ? "it" : "they"} can't be placed under
              a dataset. Runs on different dataset versions aren't comparable anyway.
            </p>
          )}
          {unreadable.length > 0 && (
            <p className="mt-0.5 text-ink-2">{plural(unreadable.length, "run has", "runs have")} files that could not be read.</p>
          )}
          <Disclosure title="Which runs" className="mt-2">
            <ul className="space-y-1 text-xs text-ink-2">
              {hidden.data.map((run) => (
                <li key={run.run_id} className="flex flex-wrap gap-x-2">
                  <span className="num">{sgt(run.evaluated_at, { seconds: false })}</span>
                  <span className="font-mono">{run.run_id}</span>
                  <span className="text-ink-3">
                    {run.reason === "dataset_missing" ? `dataset version ${shortHash(run.dataset_sha256, 8)}` : "unreadable"}
                  </span>
                </li>
              ))}
            </ul>
            <p className="mt-2 text-xs text-ink-3">
              To view one, put the dataset version it was made on back in <span className="font-mono">evaluation/datasets</span> (git history
              keeps earlier versions).
            </p>
          </Disclosure>
        </div>
      </div>
    </Card>
  );
}
