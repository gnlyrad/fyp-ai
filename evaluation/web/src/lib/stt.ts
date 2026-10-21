import { useQuery } from "@tanstack/react-query";
import { get } from "./api";

/* Types mirror the bundles written by fyp-be/evaluation-engine (see evaluation/dashboard/stt.py). */

export type SttCounts = {
  reference_words: number; substitutions: number; deletions: number; insertions: number; hits: number; errors: number; wer: number | null;
};
export type SttEntity =
  | (SttCounts & { groups: Record<string, SttCounts>; domain_terms: { reviewed: boolean; terms: number } })
  | { available: false; reason: string };
export type SttSpeaker = {
  eligible_items: number; correct_items: number; accuracy: number | null; speakers_detected?: number; label_mapping?: Record<string, string | null>;
};
export type SttConfidence = { items: number; mean: number | null; mean_correct_items?: number | null; mean_other_items?: number | null };

export type SttRunSummary = {
  run_id: string; dataset_id: string; render_id: string | null; collected_at: string | null; scored_at: string | null;
  calls_scored: number; dataset_call_count: number; normalised_wer: SttCounts; entity_wer: SttEntity; average_confidence: SttConfidence;
  speaker_attribution: SttSpeaker & { counting_unit?: string }; normalizer_version: string;
  vocabulary_name: string | null; language_model_name: string | null;
};

export type SttDataset = { dataset_id: string; label: string; run_count: number; latest_run_at: string | null; dataset_call_count: number };

export type SttOp = "equal" | "substitute" | "delete" | "insert";
export type SttPart = { ref: string | null; hyp: string | null; op: SttOp; entity: boolean };
export type SttItem = {
  item: number; content: string; confidence: number | null; speaker: string | null; start: number | null; end: number | null;
  mapped_role: string | null; eligible: boolean;
};
export type SttUnit = {
  id: number; turn_id: number; ref_text: string; hyp_text: string; op: SttOp | "mixed"; errors: number;
  substitutions: number; deletions: number; insertions: number; entity_errors: number; entity_groups: string[]; parts: SttPart[];
  ref_role: string | null; speaker_mismatch: boolean; min_confidence: number | null; low_confidence: boolean; items: SttItem[];
  scoring_rules?: string[];
  start: number | null; end: number | null; time_source: "aws" | "neighbours" | "reference_turn" | null; approximate: boolean;
};

export type SttErrorGroup = {
  op: Exclude<SttOp, "equal">; ref: string | null; hyp: string | null; count: number; entity_count: number;
  occurrences: { call_id: string; unit: number; turn_id: number; entity: boolean; context: { before: string; after: string } }[];
  examples: { ref_text: string; hyp_text: string }[];
};

export type SttRunCallRow = {
  call_id: string; wer: SttCounts; entity_wer: (SttCounts & Record<string, unknown>) | null; speaker: SttSpeaker; mean_confidence: number | null;
};

export type SttRuleApplied = Record<string, { count: number; examples: Record<string, number> }>;
export type SttChangeError = { call_id: string; turn_id: number; ref_text: string; hyp_text: string; substitutions: number; deletions: number; insertions: number };
export type SttChanges = {
  from: { normalizer_version: string; normalised_wer: SttCounts };
  to: { normalizer_version: string; normalised_wer: SttCounts };
  baseline_reproduced: boolean;
  per_call: { call_id: string; before: SttCounts; after: SttCounts }[];
  by_rule: Record<string, { errors_removed_when_all_rules_on: number; errors_removed_when_only_this_rule_on: number }>;
  removed_errors: (SttChangeError & { explained_by: string[] })[];
  remaining_errors: SttChangeError[];
  errors_that_appeared: SttChangeError[];
};

export type SttRun = {
  run_id: string; dataset_id: string; render_id: string | null; dataset_call_ids: string[]; collected_at: string | null; scored_at: string | null;
  versions: { normalizer: string; jiwer: string };
  provenance: { fyp_be_commit: string | null; fyp_be_uncommitted_changes: boolean; handler?: { git_commit: string | null; uncommitted_changes: boolean | null }; region?: string };
  transcribe_job: { language_code: string | null; settings: Record<string, unknown> | null; vocabulary_name: string | null; language_model_name: string | null };
  summary: {
    normalised_wer: SttCounts; entity_wer: SttEntity; average_confidence: SttConfidence; speaker_attribution: SttSpeaker & { counting_unit?: string };
    calls_scored: number; calls_not_completed: string[]; normalizer_version: string;
    normalizer_rules?: { enabled: string[]; spelling: { tag: string; sha256: string; entries: number }; variants: { reviewed: boolean; entries: number }; currency_table_version: string };
    scoring_rules_applied?: { reference: SttRuleApplied; hypothesis: SttRuleApplied };
  };
  changes: SttChanges | null;
  calls: SttRunCallRow[];
  errors: SttErrorGroup[];
  unnormalised: Record<string, Record<string, { count: number; examples: Record<string, number> }>>;
  audio_configured: boolean;
  summary_view: SttRunSummary;
};

export type SttCall = {
  call_id: string; run_id: string; dataset_id: string; run: SttRunSummary; normalizer_version: string;
  audio: { dataset_id: string; call_id: string; render_id: string | null; audio_sha256: string | null };
  audio_status: { available: boolean; reason: string | null };
  metrics: { wer: SttCounts; entity_wer: (SttCounts & { groups?: Record<string, SttCounts> }) | null; speaker: SttSpeaker; confidence: { items: number; sum: number } };
  turns: { turn_id: number; speaker: string; text: string; start: number | null; end: number | null }[];
  units: SttUnit[];
  entity_spans: { turn_id: number; group: string; text: string; errors: number; tokens: string[]; hypothesis_tokens: string[] }[];
  raw_reference_text: string; raw_hypothesis_text: string;
  previous_call: string | null; next_call: string | null;
};

export type SttPresentation = {
  tab_label: string;
  local_badge: { label: string; help: string };
  metrics: Record<string, { label: string; what: string; calculation: string; important?: string }>;
  coverage: Record<string, string>;
  diff: {
    rows: Record<string, string>; labels: Record<string, string>; legend: Record<string, string>; overlays: Record<string, string>;
    highlight_note: string; approximate_time: string; navigation: string; scored_totals: string; reading_key?: string;
    views?: Record<string, string>;
  };
  normaliser_note: string;
  audio: Record<string, string>;
  drills?: Record<string, string>;
  normalisation?: {
    title: string; scoring_only: string; rules: Record<string, { label: string; help: string }>; compare_intro: string; reference_words_changed: string;
    removed_title: string; removed_help: string; remaining_title: string; remaining_help: string;
  };
};

export type SttWordEntry = {
  call_id: string; unit: number; turn_id: number; ref_text: string; hyp_text: string; errors: number; ref_role: string | null;
  min_confidence: number | null; start: number | null;
  items: { content: string; confidence: number | null; speaker: string | null; mapped_role: string | null }[];
};
export type SttEntitySpan = {
  call_id: string; unit: number | null; turn_id: number; group: string; text: string; errors: number; tokens: string[]; hypothesis_tokens: string[];
};
export type SttRunWords = {
  confidence_buckets: { from: number; to: number; matched: number; other: number }[];
  low_confidence: SttWordEntry[];  // every unit below 0.9; the page narrows it
  speaker_mismatch: SttWordEntry[];
  entity_spans: SttEntitySpan[];
};

const enc = encodeURIComponent;

export const useSttDatasets = () => useQuery({ queryKey: ["stt", "datasets"], queryFn: () => get<SttDataset[]>("/stt/datasets") });
export const useSttDataset = (datasetId: string) =>
  useQuery({ queryKey: ["stt", "dataset", datasetId], queryFn: () => get<{ dataset: SttDataset; runs: SttRunSummary[] }>(`/stt/datasets/${enc(datasetId)}`) });
export const useSttRun = (runId: string) => useQuery({ queryKey: ["stt", "run", runId], queryFn: () => get<SttRun>(`/stt/runs/${enc(runId)}`) });
export const useSttRunWords = (runId: string, enabled: boolean) =>
  useQuery({ queryKey: ["stt", "words", runId], queryFn: () => get<SttRunWords>(`/stt/runs/${enc(runId)}/words`), enabled });
export const useSttCall = (runId: string, callId: string) =>
  useQuery({ queryKey: ["stt", "call", runId, callId], queryFn: () => get<SttCall>(`/stt/runs/${enc(runId)}/calls/${enc(callId)}`) });

export const sttPaths = {
  home: () => "/stt",
  dataset: (datasetId: string) => `/stt/d/${enc(datasetId)}`,
  run: (datasetId: string, runId: string) => `/stt/d/${enc(datasetId)}/r/${enc(runId)}`,
  call: (datasetId: string, runId: string, callId: string, unit?: number) =>
    `/stt/d/${enc(datasetId)}/r/${enc(runId)}/c/${enc(callId)}${unit === undefined ? "" : `?unit=${unit}`}`,
  audio: (runId: string, callId: string) => `/api/stt/runs/${enc(runId)}/calls/${enc(callId)}/audio`,
};
