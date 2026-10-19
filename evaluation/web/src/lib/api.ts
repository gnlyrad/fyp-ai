import { useQuery } from "@tanstack/react-query";

/* Types mirror evaluation/dashboard/api.py. Summary nodes stay loosely typed
   because the dashboard reads them through metric summary_paths. */

export type Ratio = { value: number | null; percentage: number | null; numerator: number; denominator: number };
export type SummaryNode = Record<string, any>;
export type Tone = "good" | "bad" | "warn" | "neutral";

export type MetricDefinition = {
  id: string;
  label: string;
  role: string;
  applies_to: string;
  population: string;
  summary_path: string;
  formula: string;
  interpretation: string;
  watch_out?: string;
  /** summary paths of the same metric split by whether the ground truth expected evidence */
  breakdowns?: { non_empty_gold: string; empty_gold: string };
};

export type MetricWording = {
  label: string;
  audience: "summary" | "developer";
  what: string;
  calculation: string;
  important?: string;
};

export type Labelled = { label: string; description?: string };

export type Presentation = {
  families: Record<string, Labelled>;
  /** Readable names for the internal extraction field identifiers. */
  fields: Record<string, { label: string }>;
  metrics: Record<string, MetricWording>;
  statuses: Record<string, Record<string, Labelled>>;
  confusion_categories: Record<string, Labelled>;
  run_status: Record<string, Labelled>;
  evidence_parts: Record<string, any>;
  operations: Record<string, MetricWording>;
  error_categories: Record<string, Labelled>;
};

export type Drill = { label: string; count_path: string | null; filter: Record<string, string> };

export type Meta = {
  default_source: Source;
  sources: Source[];
  definitions: { metrics: MetricDefinition[] } & Record<string, any>;
  presentation: Presentation;
  drills: Record<string, Drill>;
};

export type Source = "shared" | "local";

export type Dataset = {
  dataset_id: string;
  name: string;
  label: string;
  call_ids: string[];
  call_count: number;
  question_count: number;
  compliance_questions: number;
  extraction_questions: number;
  dataset_sha256: string;
  checklist_sha256: string | null;
  manifest_sha256: string | null;
  format: string;
  run_count?: number;
  latest_run_at?: string | null;
};

export type RunStatus = {
  raw: string;
  label: string;
  description: string;
  tone: Tone;
  prediction_failures: number;
  pipeline_warnings: number | null;
  detail: string;
};

export type Operations = {
  calls: number;
  total_latency_ms: number | null;
  avg_call_latency_ms: number | null;
  tokens: { input_tokens: number; output_tokens: number; total_tokens: number; reasoning_tokens: number } | null;
  cost: {
    amount: number | null;
    currency: string;
    per_call: number | null;
    unpriced_models: string[];
    prices: Record<string, { input: number; output: number; cached_input: number | null; source?: string; catalog?: string; as_of?: string }>;
  } | null;
};

export type RunSummary = {
  run_id: string;
  dataset_id: string;
  evaluated_at: string | null;
  is_complete: boolean;
  status: RunStatus;
  graph_commit: string | null;
  graph_worktree_dirty: boolean;
  chat_model_id: string | null;
  run_kind: string | null;
  metrics: Record<string, Ratio | null>;
  operations: Operations;
};

export type Row = {
  call_id: string;
  question_id: string;
  family: "compliance" | "extraction";
  question: string | null;
  field: string | null;
  requirement_type: string | null;
  execution_status: string;
  failure_category: string | null;
  native_verdict: string | null;
  gold_verdict: string | null;
  predicted_verdict: string | null;
  compliance_outcome_status: string | null;
  confusion_category: string | null;
  binary_correct: number | null;
  breach_outcome: string | null;
  expected_value: string | null;
  value_type: string | null;
  gold_value_status: string | null;
  evaluator_support: string | null;
  extraction_evidence_status: string | null;
  extraction_evidence_coverage: number | null;
  /** CITED / MISSED / NONE_EXPECTED for extraction checks; null for compliance and failed checks. */
  extraction_turn_status: string | null;
  gold_turns: number[];
  predicted_turns: number[];
  evidence_status: string | null;
  evidence_precision: number | null;
  evidence_recall: number | null;
  evidence_exact_match: number | null;
  invalid_reference_count: number;
  outcome: { label: string; tone: Tone };
};

export type Problem = {
  family: string;
  label: string;
  count: number;
  target: "cases" | "diagnostics";
  drill?: string;
  category?: string;
};

export type ChecklistItem = {
  id: string;
  question: string;
  type: string;
  evaluation_family: "compliance" | "extraction";
  field?: string;
  requirement_type?: string;
};

export type RunDetail = {
  run: RunSummary & { completed_at: string | null; source: string; metadata: Record<string, any> };
  dataset: { dataset_id: string; label: string; name: string };
  checklist: ChecklistItem[];
  summary: { overall: SummaryNode; by_call_id: Record<string, SummaryNode>; by_question_id: Record<string, SummaryNode> };
  rows: Row[];
  problems: Problem[];
};

export type Turn = { id: number; speaker: string; text: string; start_time: string | null; expected: boolean; selected: boolean };

export type ResultDetail = {
  row: Row;
  question: ChecklistItem;
  turns: Turn[];
  explanation: string;
  selected_quotes: string[];
  reasoning: string | null;
  failure_message: string | null;
  prediction: Record<string, any>;
  score: Record<string, any>;
  ground_truth: Record<string, any>;
};

export type EvidenceTurn = { id: number; speaker: string; text: string };

export type EvidenceEntry = {
  call_id: string;
  question_id: string;
  family: "compliance" | "extraction";
  question: string | null;
  execution_status: string;
  native_verdict: string | null;
  evidence_status: string | null;
  precision: number | null;
  recall: number | null;
  exact: number | null;
  gold_turns: number[];
  predicted_turns: number[];
  missed: EvidenceTurn[];
  extra: EvidenceTurn[];
  invalid_reference_count: number;
  unresolved_references: string[];
  explanation: string;
  outcome: { label: string; tone: Tone } | null;
};

export type DiagnosticError = {
  index: number;
  category: string;
  label: string;
  description: string | null;
  call_id: string | null;
  question_id: string | null;
  message: string | null;
  linkable: boolean;
  payload: Record<string, any>;
};

export type UnlinkedRun = {
  run_id: string;
  evaluated_at: string | null;
  dataset_sha256: string | null;
  reason: "dataset_missing" | "unreadable";
};

export type Comparison = {
  earlier: RunSummary;
  later: RunSummary;
  comparable: boolean;
  incompatibilities: string[];
  unverifiable_run_ids: string[];
  metrics: { metric_id: string; earlier: Ratio | null; later: Ratio | null; delta_pp: number | null }[];
  operations: { id: string; unit: "ms" | "tokens" | "currency"; earlier: number | null; later: number | null; delta: number | null }[];
};

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

/** FastAPI sends a string for most errors but a list of {loc, msg} objects for validation errors. */
function describe(detail: unknown): string | undefined {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => (item && typeof item === "object" ? `${(item as any).loc?.slice(1).join(".") ?? "request"}: ${(item as any).msg ?? "invalid"}` : String(item)))
      .join("; ");
  }
  return undefined;
}

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`/api${path}`);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = describe((await response.json()).detail) ?? detail;
    } catch {
      /* non-JSON error body */
    }
    if (response.status === 422) {
      detail += " (the dashboard page and the server are out of step: restart serve_dashboard.py)";
    }
    throw new ApiError(response.status, detail);
  }
  return response.json() as Promise<T>;
}

const enc = encodeURIComponent;

export const useMeta = () => useQuery({ queryKey: ["meta"], queryFn: () => get<Meta>("/meta"), staleTime: Infinity });

export const useDatasets = (source: Source) =>
  useQuery({ queryKey: ["datasets", source], queryFn: () => get<Dataset[]>(`/sources/${source}/datasets`) });

export const useDataset = (source: Source, datasetId: string) =>
  useQuery({
    queryKey: ["dataset", source, datasetId],
    queryFn: () => get<{ dataset: Dataset; runs: RunSummary[] }>(`/sources/${source}/datasets/${enc(datasetId)}`),
  });

export const useRun = (source: Source, runId: string) =>
  useQuery({ queryKey: ["run", source, runId], queryFn: () => get<RunDetail>(`/sources/${source}/runs/${enc(runId)}`) });

export const useResult = (source: Source, runId: string, callId: string, questionId: string) =>
  useQuery({
    queryKey: ["result", source, runId, callId, questionId],
    queryFn: () => get<ResultDetail>(`/sources/${source}/runs/${enc(runId)}/results/${enc(callId)}/${enc(questionId)}`),
  });

export const useDiagnostics = (source: Source, runId: string) =>
  useQuery({
    queryKey: ["diagnostics", source, runId],
    queryFn: () => get<{ errors: DiagnosticError[]; counts: Record<string, number> }>(`/sources/${source}/runs/${enc(runId)}/diagnostics`),
  });

export const useUnlinkedRuns = (source: Source) =>
  useQuery({ queryKey: ["unlinked", source], queryFn: () => get<UnlinkedRun[]>(`/sources/${source}/unlinked-runs`) });

export const useEvidence = (source: Source, runId: string) =>
  useQuery({
    queryKey: ["evidence", source, runId],
    queryFn: () => get<{ entries: EvidenceEntry[] }>(`/sources/${source}/runs/${enc(runId)}/evidence`),
  });

export const useComparison = (source: Source, a: string, b: string) =>
  useQuery({
    queryKey: ["compare", source, a, b],
    queryFn: () => get<Comparison>(`/sources/${source}/compare?a=${enc(a)}&b=${enc(b)}`),
    enabled: Boolean(a && b),
  });
