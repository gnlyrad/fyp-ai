from pathlib import Path
from types import SimpleNamespace

import pytest

from dashboard.repository import RepositoryError
from dashboard.supabase_repository import SupabaseEvaluationRepository
from fyp_evaluation_engine.publication import PublicationError, SupabasePublisher

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_shared_repository_requires_only_project_values(monkeypatch):
    for name in (
        "SUPABASE_URL",
        "SUPABASE_PUBLISHABLE_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RepositoryError, match="Shared source requires"):
        SupabaseEvaluationRepository.from_environment()


def test_shared_repository_does_not_require_or_attempt_auth(monkeypatch):
    marker = object()
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_example")
    monkeypatch.delenv("SUPABASE_DASHBOARD_EMAIL", raising=False)
    monkeypatch.delenv("SUPABASE_DASHBOARD_PASSWORD", raising=False)
    monkeypatch.setattr("dashboard.supabase_repository.create_client", lambda *_: marker)
    repository = SupabaseEvaluationRepository.from_environment()
    assert repository.client is marker


def test_publisher_needs_only_project_values_and_attempts_no_login(monkeypatch):
    for name in ("SUPABASE_URL", "SUPABASE_PUBLISHABLE_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(PublicationError, match="SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY"):
        SupabasePublisher.from_environment()

    class Client:
        storage = SimpleNamespace(from_=lambda bucket: bucket)

        @property
        def auth(self):
            raise AssertionError("Publishing must not sign in")

    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_example")
    monkeypatch.setattr("supabase.create_client", lambda *_: Client())
    assert isinstance(SupabasePublisher.from_environment(), SupabasePublisher)


def test_migrations_let_the_key_publish_but_never_edit_or_delete():
    directory = REPOSITORY_ROOT / "supabase" / "migrations"
    migrations = "\n".join(path.read_text(encoding="utf-8") for path in sorted(directory.glob("*.sql"))).lower()
    assert "enable row level security" in migrations
    assert "grant select on table" in migrations
    assert "public = true" not in migrations
    # Key holders may insert and finalize ...
    assert "for insert to anon" in migrations
    assert "grant update (publication_state) on table" in migrations
    # ... but published records stay immutable: no delete, no broad update, no service role.
    for forbidden in ("for delete to anon", "for all to anon", "grant delete", "grant all to anon", "service_role"):
        assert forbidden not in migrations
    keyholder = (directory / "20261004000000_allow_keyholder_publication.sql").read_text(encoding="utf-8").lower()
    assert "using (publication_state = 'publishing')" in keyholder
    assert "with check (publication_state = 'published')" in keyholder


def test_shared_repository_paginates_large_result_sets():
    class Query:
        def __init__(self):
            self.rows = [{"row": index} for index in range(2001)]
            self.window = (0, 999)

        def range(self, start, end):
            self.window = (start, end)
            return self

        def execute(self):
            start, end = self.window
            return SimpleNamespace(data=self.rows[start:end + 1])

    rows = SupabaseEvaluationRepository._all_rows(Query())
    assert len(rows) == 2001
    assert rows[-1] == {"row": 2000}
