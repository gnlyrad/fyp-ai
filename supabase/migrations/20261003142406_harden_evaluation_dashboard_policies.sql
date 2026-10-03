-- Cache JWT lookup once per statement in RLS policies and remove direct API access
-- to Supabase's automatic-RLS trigger helper.
alter policy evaluation_datasets_read on public.evaluation_datasets
    using (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
alter policy evaluation_dataset_calls_read on public.evaluation_dataset_calls
    using (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
alter policy evaluation_checklist_items_read on public.evaluation_checklist_items
    using (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
alter policy evaluation_runs_read on public.evaluation_runs
    using (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
alter policy evaluation_run_metrics_read on public.evaluation_run_metrics
    using (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
alter policy evaluation_run_results_read on public.evaluation_run_results
    using (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
alter policy evaluation_run_errors_read on public.evaluation_run_errors
    using (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');

alter policy evaluation_datasets_publish on public.evaluation_datasets
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
alter policy evaluation_dataset_calls_publish on public.evaluation_dataset_calls
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
alter policy evaluation_checklist_items_publish on public.evaluation_checklist_items
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
alter policy evaluation_runs_publish on public.evaluation_runs
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
alter policy evaluation_run_metrics_publish on public.evaluation_run_metrics
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
alter policy evaluation_run_results_publish on public.evaluation_run_results
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
alter policy evaluation_run_errors_publish on public.evaluation_run_errors
    with check (((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');

alter policy evaluation_datasets_finalize on public.evaluation_datasets
    using (publication_state = 'publishing'
        and ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true')
    with check (publication_state = 'published'
        and ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');
alter policy evaluation_runs_finalize on public.evaluation_runs
    using (publication_state = 'publishing'
        and ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true')
    with check (publication_state = 'published'
        and ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true');

alter policy evaluation_artifacts_read on storage.objects
    using (bucket_id = 'evaluation-artifacts' and (
        ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_reader') = 'true'
        or ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
    ));
alter policy evaluation_artifacts_publish on storage.objects
    with check (
        bucket_id = 'evaluation-artifacts'
        and ((select auth.jwt()) -> 'app_metadata' ->> 'evaluation_publisher') = 'true'
        and (storage.foldername(name))[1] in ('datasets', 'runs')
    );

do $$
begin
    if to_regprocedure('public.rls_auto_enable()') is not null then
        execute 'revoke execute on function public.rls_auto_enable() from public, anon, authenticated';
    end if;
end
$$;
