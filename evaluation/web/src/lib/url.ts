import { useCallback } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import type { Source } from "./api";

/** Every drill-down lives in the URL, so it is linkable, restorable and Back-able. */
export function useQueryState() {
  const [params, setParams] = useSearchParams();
  const update = useCallback(
    (changes: Record<string, string | null | undefined>, { replace = false } = {}) => {
      setParams(
        (current) => {
          const next = new URLSearchParams(current);
          for (const [key, value] of Object.entries(changes)) {
            if (value === null || value === undefined || value === "") next.delete(key);
            else next.set(key, value);
          }
          return next;
        },
        { replace },
      );
    },
    [setParams],
  );
  return [params, update] as const;
}

export function useSource(): Source {
  const { source } = useParams();
  return source === "local" ? "local" : "shared";
}

export const paths = {
  datasets: (source: Source) => `/${source}`,
  dataset: (source: Source, datasetId: string) => `/${source}/d/${encodeURIComponent(datasetId)}`,
  compare: (source: Source, datasetId: string, a: string, b: string) =>
    `${paths.dataset(source, datasetId)}/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`,
  run: (source: Source, datasetId: string, runId: string) => `${paths.dataset(source, datasetId)}/r/${encodeURIComponent(runId)}`,
  result: (source: Source, datasetId: string, runId: string, callId: string, questionId: string) =>
    `${paths.run(source, datasetId, runId)}/result/${encodeURIComponent(callId)}/${encodeURIComponent(questionId)}`,
};

/** Router state handed to the Individual Result page so prev/next and Back follow the list it came from. */
export type ResultNavState = { from: string; list: [string, string][]; listLabel: string };
