-- DECISION: anyone holding the project's publishable key may publish evaluation runs.
-- There is no publisher login any more (the earlier evaluation_publisher claim policies
-- stay in place but are no longer required).
--
-- What this opens: INSERT of new rows/objects, and moving a row from 'publishing' to
-- 'published'. What stays closed: DELETE, every other UPDATE (so published records are
-- immutable through the key), and the service-role key is never used. The publisher is
-- also idempotent, so repeating an identical publish is a no-op and a conflicting one
-- is rejected client-side.
--
-- Trade-off: the key lives in .env files, so a leaked key can add junk records or
-- finalize an interrupted publication. Rotate the publishable key if that happens.

grant insert on table
    public.evaluation_datasets,
    public.evaluation_dataset_calls,
    public.evaluation_checklist_items,
    public.evaluation_runs,
    public.evaluation_run_metrics,
    public.evaluation_run_results,
    public.evaluation_run_errors
to anon;

grant update (publication_state) on table
    public.evaluation_datasets,
    public.evaluation_runs
to anon;

create policy evaluation_datasets_keyholder_insert on public.evaluation_datasets
    for insert to anon with check (publication_state = 'publishing');
create policy evaluation_runs_keyholder_insert on public.evaluation_runs
    for insert to anon with check (publication_state = 'publishing');
create policy evaluation_dataset_calls_keyholder_insert on public.evaluation_dataset_calls
    for insert to anon with check (true);
create policy evaluation_checklist_items_keyholder_insert on public.evaluation_checklist_items
    for insert to anon with check (true);
create policy evaluation_run_metrics_keyholder_insert on public.evaluation_run_metrics
    for insert to anon with check (true);
create policy evaluation_run_results_keyholder_insert on public.evaluation_run_results
    for insert to anon with check (true);
create policy evaluation_run_errors_keyholder_insert on public.evaluation_run_errors
    for insert to anon with check (true);

create policy evaluation_datasets_keyholder_finalize on public.evaluation_datasets
    for update to anon
    using (publication_state = 'publishing')
    with check (publication_state = 'published');
create policy evaluation_runs_keyholder_finalize on public.evaluation_runs
    for update to anon
    using (publication_state = 'publishing')
    with check (publication_state = 'published');

-- The publisher reads back unfinished rows to resume an interrupted publication, so the
-- read policies no longer hide 'publishing' rows. The dashboard queries always filter on
-- publication_state = 'published', so half-published runs still never appear there.
alter policy evaluation_datasets_read on public.evaluation_datasets to anon, authenticated using (true);
alter policy evaluation_dataset_calls_read on public.evaluation_dataset_calls to anon, authenticated using (true);
alter policy evaluation_checklist_items_read on public.evaluation_checklist_items to anon, authenticated using (true);
alter policy evaluation_runs_read on public.evaluation_runs to anon, authenticated using (true);
alter policy evaluation_run_metrics_read on public.evaluation_run_metrics to anon, authenticated using (true);
alter policy evaluation_run_results_read on public.evaluation_run_results to anon, authenticated using (true);
alter policy evaluation_run_errors_read on public.evaluation_run_errors to anon, authenticated using (true);

-- Storage: new objects only (no update/delete policy, so existing files cannot be
-- replaced). Read is needed so the publisher can verify an existing object byte-for-byte.
create policy evaluation_artifacts_keyholder_insert on storage.objects
    for insert to anon
    with check (
        bucket_id = 'evaluation-artifacts'
        and (storage.foldername(name))[1] in ('datasets', 'runs')
    );
create policy evaluation_artifacts_keyholder_read on storage.objects
    for select to anon
    using (bucket_id = 'evaluation-artifacts');
