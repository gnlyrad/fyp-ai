# Shared evaluation storage

The migrations in this directory provision the shared evaluation store:

- immutable dataset snapshots, calls and checklist definitions;
- queryable run, metric, result and error records;
- the private `evaluation-artifacts` Storage bucket for complete source artifacts;
- public read access to finalized synthetic evaluation records through the publishable key;
- publication through the publishable key (insert and finalize only; no edits or deletes).

No service-role key is used by the dashboard or uploader. Anyone holding the project URL
and publishable key can both view the dashboard and run `upload_run.py`: there are no
logins and no publisher accounts. RLS stays enabled as a guardrail, so that key can
insert new records and move a run from `publishing` to `published`, but cannot update
anything else or delete anything. Treat the key as a shared team secret and rotate it in
Supabase if it leaks.

## One-time project setup

1. Apply every file in `supabase/migrations/` in filename order, including
   `20261004000000_allow_keyholder_publication.sql` (paste it into the Supabase SQL editor
   or run `supabase db push`).
2. Give teammates the project URL and publishable key. Nothing else is required.

## Local environment

Create an uncommitted `.env` based on `.env.example`:

```dotenv
SUPABASE_URL=
SUPABASE_PUBLISHABLE_KEY=
EVALUATION_DEFAULT_SOURCE=Shared
```

That is the complete configuration for both viewing and publishing.

## Publication lifecycle

`evaluation/scripts/upload_run.py --run-id <RUN_ID>` processes only that run:

```text
validate complete artifacts and full dataset coverage
→ compare the run ID and content fingerprint
→ upload/reuse the immutable dataset snapshot
→ upload/reuse private Storage objects
→ insert queryable records
→ mark the dataset and run published
```

Rows in the temporary `publishing` state allow an interrupted publication to resume
safely. The dashboard only ever queries `published` rows, so unfinished runs never
appear in it. Storage artifacts can be added and read through the key, never replaced
or deleted.
