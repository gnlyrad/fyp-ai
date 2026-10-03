-- Shared, immutable evaluation history. All dashboard reads are synthetic-data-only.
create table public.evaluation_datasets (
    dataset_sha256 text primary key check (dataset_sha256 ~ '^[0-9a-f]{64}$'),
    dataset_name text not null,
    checklist_sha256 text not null check (checklist_sha256 ~ '^[0-9a-f]{64}$'),
    manifest_sha256 text not null check (manifest_sha256 ~ '^[0-9a-f]{64}$'),
    dataset_format text not null,
    call_ids jsonb not null check (jsonb_typeof(call_ids) = 'array'),
    question_count integer not null check (question_count > 0),
    storage_prefix text not null unique,
    manifest jsonb not null,
    publication_state text not null default 'publishing'
        check (publication_state in ('publishing', 'published')),
    created_at timestamptz not null default now()
);

create table public.evaluation_dataset_calls (
    dataset_sha256 text not null references public.evaluation_datasets(dataset_sha256),
    call_id text not null,
    transcript jsonb not null,
    speaker_roles jsonb not null,
    ground_truth jsonb not null,
    primary key (dataset_sha256, call_id)
);

create table public.evaluation_checklist_items (
    dataset_sha256 text not null references public.evaluation_datasets(dataset_sha256),
    question_id text not null,
    question text not null,
    evaluation_family text not null check (evaluation_family in ('compliance', 'extraction')),
    question_type text not null,
    requirement_type text,
    field text,
    definition jsonb not null,
    primary key (dataset_sha256, question_id)
);

create table public.evaluation_runs (
    run_id text primary key,
    dataset_sha256 text not null references public.evaluation_datasets(dataset_sha256),
    artifact_fingerprint_sha256 text not null check (artifact_fingerprint_sha256 ~ '^[0-9a-f]{64}$'),
    started_at timestamptz,
    completed_at timestamptz,
    execution_status text not null check (execution_status in ('completed', 'completed_with_errors')),
    run_kind text not null,
    artifact_schema_version text not null,
    scorer_version text not null,
    metric_definitions_version text not null,
    metric_definitions_sha256 text not null check (metric_definitions_sha256 ~ '^[0-9a-f]{64}$'),
    checklist_sha256 text not null check (checklist_sha256 ~ '^[0-9a-f]{64}$'),
    graph_commit text,
    git_commit_author text,
    git_commit_timestamp timestamptz,
    graph_worktree_dirty boolean,
    total_latency_ms double precision,
    selected_call_ids jsonb not null check (jsonb_typeof(selected_call_ids) = 'array'),
    storage_prefix text not null unique,
    metadata jsonb not null,
    summary jsonb not null,
    publication_state text not null default 'publishing'
        check (publication_state in ('publishing', 'published')),
    published_at timestamptz not null default now()
);

create index evaluation_runs_dataset_started_idx
    on public.evaluation_runs (dataset_sha256, started_at desc);

create table public.evaluation_run_metrics (
    run_id text not null references public.evaluation_runs(run_id),
    scope_type text not null check (scope_type in ('overall', 'call', 'question')),
    scope_id text not null default '',
    evaluation_family text not null check (evaluation_family in ('compliance', 'extraction', 'both')),
    metric_id text not null,
    numerator double precision,
    denominator double precision,
    value double precision,
    percentage double precision,
    payload jsonb not null,
    primary key (run_id, scope_type, scope_id, metric_id)
);

create index evaluation_run_metrics_lookup_idx
    on public.evaluation_run_metrics (run_id, scope_type, evaluation_family);

create table public.evaluation_run_results (
    run_id text not null references public.evaluation_runs(run_id),
    call_id text not null,
    question_id text not null,
    evaluation_family text not null check (evaluation_family in ('compliance', 'extraction')),
    execution_status text not null,
    failure_category text,
    gold_verdict text,
    predicted_verdict text,
    compliance_outcome_status text,
    confusion_category text,
    breach_outcome text,
    expected_value text,
    gold_value_status text,
    extraction_evidence_status text,
    evidence_precision double precision,
    evidence_recall double precision,
    evidence_exact_set_match double precision,
    result_row jsonb not null,
    score jsonb not null,
    prediction jsonb not null,
    primary key (run_id, call_id, question_id)
);

create index evaluation_run_results_call_idx
    on public.evaluation_run_results (run_id, call_id);
create index evaluation_run_results_question_idx
    on public.evaluation_run_results (run_id, question_id);
create index evaluation_run_results_family_idx
    on public.evaluation_run_results (run_id, evaluation_family);

create table public.evaluation_run_errors (
    run_id text not null references public.evaluation_runs(run_id),
    error_index integer not null,
    call_id text,
    question_id text,
    category text,
    message text,
    payload jsonb not null,
    primary key (run_id, error_index)
);

alter table public.evaluation_datasets enable row level security;
alter table public.evaluation_dataset_calls enable row level security;
alter table public.evaluation_checklist_items enable row level security;
alter table public.evaluation_runs enable row level security;
alter table public.evaluation_run_metrics enable row level security;
alter table public.evaluation_run_results enable row level security;
alter table public.evaluation_run_errors enable row level security;

revoke all on table
    public.evaluation_datasets,
    public.evaluation_dataset_calls,
    public.evaluation_checklist_items,
    public.evaluation_runs,
    public.evaluation_run_metrics,
    public.evaluation_run_results,
    public.evaluation_run_errors
from anon, authenticated;

grant select on table
    public.evaluation_datasets,
    public.evaluation_dataset_calls,
    public.evaluation_checklist_items,
    public.evaluation_runs,
    public.evaluation_run_metrics,
    public.evaluation_run_results,
    public.evaluation_run_errors
to authenticated;

grant insert on table
    public.evaluation_datasets,
    public.evaluation_dataset_calls,
    public.evaluation_checklist_items,
    public.evaluation_runs,
    public.evaluation_run_metrics,
    public.evaluation_run_results,
    public.evaluation_run_errors
to authenticated;

grant update (publication_state) on table
    public.evaluation_datasets,
    public.evaluation_runs
to authenticated;

create policy evaluation_datasets_read on public.evaluation_datasets
    for select to authenticated using (
        ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );
create policy evaluation_dataset_calls_read on public.evaluation_dataset_calls
    for select to authenticated using (
        ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );
create policy evaluation_checklist_items_read on public.evaluation_checklist_items
    for select to authenticated using (
        ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );
create policy evaluation_runs_read on public.evaluation_runs
    for select to authenticated using (
        ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );
create policy evaluation_run_metrics_read on public.evaluation_run_metrics
    for select to authenticated using (
        ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );
create policy evaluation_run_results_read on public.evaluation_run_results
    for select to authenticated using (
        ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );
create policy evaluation_run_errors_read on public.evaluation_run_errors
    for select to authenticated using (
        ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );

-- Publisher authorization is controlled only by immutable app_metadata, never user_metadata.
create policy evaluation_datasets_publish on public.evaluation_datasets
    for insert to authenticated
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
create policy evaluation_dataset_calls_publish on public.evaluation_dataset_calls
    for insert to authenticated
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
create policy evaluation_checklist_items_publish on public.evaluation_checklist_items
    for insert to authenticated
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
create policy evaluation_runs_publish on public.evaluation_runs
    for insert to authenticated
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
create policy evaluation_run_metrics_publish on public.evaluation_run_metrics
    for insert to authenticated
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
create policy evaluation_run_results_publish on public.evaluation_run_results
    for insert to authenticated
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
create policy evaluation_run_errors_publish on public.evaluation_run_errors
    for insert to authenticated
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');

create policy evaluation_datasets_finalize on public.evaluation_datasets
    for update to authenticated
    using (
        publication_state = 'publishing'
        and ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    )
    with check (
        publication_state = 'published'
        and ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );
create policy evaluation_runs_finalize on public.evaluation_runs
    for update to authenticated
    using (
        publication_state = 'publishing'
        and ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    )
    with check (
        publication_state = 'published'
        and ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );

insert into storage.buckets (id, name, public)
values ('evaluation-artifacts', 'evaluation-artifacts', false)
on conflict (id) do nothing;

create policy evaluation_artifacts_read
    on storage.objects for select
    to authenticated
    using (
        bucket_id = 'evaluation-artifacts'
        and (
            ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
            or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
        )
    );

create policy evaluation_artifacts_publish
    on storage.objects for insert
    to authenticated
    with check (
        bucket_id = 'evaluation-artifacts'
        and ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
        and (storage.foldername(name))[1] in ('datasets', 'runs')
    );
