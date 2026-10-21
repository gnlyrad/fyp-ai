# Task scope and ownership

These instructions apply throughout this fyp-ai repository.

## Default role: production NLP contributor

Unless the current task explicitly assigns evaluation, dataset, publication,
dashboard or audio/STT maintenance, work as a production NLP contributor.
Requests to improve accuracy, recall, coverage, cost, speed or reliability do
not authorize changes to those separately maintained areas.

Allowed within the assigned production task:
- NLP implementation in langgraph-backend/compliance_pipeline/.
- Model, settings and retrieval implementation in langchain-backend/langchain_backend/.
- Corresponding production-only tests in both backends.
- Necessary production configuration examples, with effective settings reported.

Protected during production NLP tasks:
- evaluation/ in full: input projection, loaders, runners, scoring, adapters,
  metrics, tests, dashboard, scripts, publication and historical artifacts.
- All benchmark datasets, gold labels, evidence, manifests and fingerprints.
- supabase/, database/storage permissions and publication workflows.
- Audio/STT functionality and integration, including inside otherwise allowed folders.
- Retrieval corpus documents, generated indexes, secrets and unrelated configuration.

Reading protected code to understand interfaces is allowed. If integration
requires a protected change, describe the exact proposed patch for its maintainer;
do not implement it or bypass the boundary with wrappers or side channels.
Do not alter these instructions to expand the task's permissions.

## Explicit maintainer tasks

Protected changes are permitted when explicitly assigned by the user, only within
the authorized scope. Keep them separate from production NLP commits. Permission
for one integration change does not authorize other evaluator or dataset changes.
Preserve unrelated and uncommitted work; do not broadly restore files or rewrite
shared history. Coordinate changes to shared files to minimize merge conflicts.

## Production and benchmark integrity

Implement model calls, question grouping, verification and bounded runtime/output
retries in production. Preserve the existing public input contract unless a
contract change is explicitly assigned. Never read benchmark files from production
to recover missing input metadata. Never use gold answers, gold evidence, challenge
tags or evaluation scores as prediction inputs or retry/selection feedback.

Do not change gold labels to improve scores or edit historical run artifacts.
Offline tests are allowed; paid inference, benchmark execution, rescoring,
publication, cloud deletion, corpus ingestion and index rebuilds need explicit
task authorization. Report changed files, offline checks and effective production
configuration. Do not claim a merge or rebase is guaranteed conflict-free.
