-- The dashboard is intentionally public for the synthetic benchmark data.
-- Keep RLS enabled and expose only finalized rows to the anon role used by a
-- publishable key. Publication remains restricted to approved Auth users.
grant select on table
    public.evaluation_datasets,
    public.evaluation_dataset_calls,
    public.evaluation_checklist_items,
    public.evaluation_runs,
    public.evaluation_run_metrics,
    public.evaluation_run_results,
    public.evaluation_run_errors
to anon;

alter policy evaluation_datasets_read on public.evaluation_datasets
    to anon, authenticated
    using (
        publication_state = 'published'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );

alter policy evaluation_dataset_calls_read on public.evaluation_dataset_calls
    to anon, authenticated
    using (
        exists (
            select 1
            from public.evaluation_datasets dataset
            where dataset.dataset_sha256 = evaluation_dataset_calls.dataset_sha256
              and dataset.publication_state = 'published'
        )
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );

alter policy evaluation_checklist_items_read on public.evaluation_checklist_items
    to anon, authenticated
    using (
        exists (
            select 1
            from public.evaluation_datasets dataset
            where dataset.dataset_sha256 = evaluation_checklist_items.dataset_sha256
              and dataset.publication_state = 'published'
        )
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );

alter policy evaluation_runs_read on public.evaluation_runs
    to anon, authenticated
    using (
        publication_state = 'published'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );

alter policy evaluation_run_metrics_read on public.evaluation_run_metrics
    to anon, authenticated
    using (
        exists (
            select 1
            from public.evaluation_runs run
            where run.run_id = evaluation_run_metrics.run_id
              and run.publication_state = 'published'
        )
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );

alter policy evaluation_run_results_read on public.evaluation_run_results
    to anon, authenticated
    using (
        exists (
            select 1
            from public.evaluation_runs run
            where run.run_id = evaluation_run_results.run_id
              and run.publication_state = 'published'
        )
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );

alter policy evaluation_run_errors_read on public.evaluation_run_errors
    to anon, authenticated
    using (
        exists (
            select 1
            from public.evaluation_runs run
            where run.run_id = evaluation_run_errors.run_id
              and run.publication_state = 'published'
        )
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );

-- Storage contains the source artifact bundles and stays private. The dashboard
-- uses the queryable JSONB records above, so public Storage access is unnecessary.
alter policy evaluation_artifacts_read on storage.objects
    to authenticated
    using (
        bucket_id = 'evaluation-artifacts'
        and ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    );
