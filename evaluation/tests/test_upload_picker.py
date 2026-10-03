"""The interactive run picker in scripts/upload_run.py."""
import importlib.util
import sys
from pathlib import Path

import pytest

from dashboard.domain import RunRecord

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "upload_run.py"
spec = importlib.util.spec_from_file_location("upload_run_script", SCRIPT)
upload_run = importlib.util.module_from_spec(spec)
sys.modules["upload_run_script"] = upload_run
spec.loader.exec_module(upload_run)


def _run(run_id, started, complete=True):
    return {"dataset": "pilot", "run": RunRecord(run_id=run_id, dataset_id="d", source="Local", is_complete=complete,
                                                  metadata={"started_at": started}, summary={})}


class _Publisher:
    def __init__(self, published=()):
        self.published = set(published)

    def published_run_ids(self):
        return self.published


@pytest.fixture
def runs(monkeypatch):
    items = [_run("run-new", "2026-10-04T01:00:00Z"), _run("run-partial", "2026-10-03T01:00:00Z", complete=False),
             _run("run-old", "2026-10-02T01:00:00Z")]
    monkeypatch.setattr(upload_run, "_local_runs", lambda: items)
    return items


def _answers(monkeypatch, *values):
    pending = iter(values)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(pending))


def test_picker_returns_the_chosen_run_and_shows_status_in_singapore_time(monkeypatch, runs, capsys):
    _answers(monkeypatch, "3")
    assert upload_run.choose_run(_Publisher(published={"run-old"})) == "run-old"
    shown = capsys.readouterr().out
    assert "04 Oct 2026 09:00 AM SGT" in shown  # 01:00 UTC
    assert "(ready)" in shown and "(partial run, cannot be published)" in shown and "(already published)" in shown


def test_picker_refuses_partial_runs_and_reprompts(monkeypatch, runs, capsys):
    _answers(monkeypatch, "2", "99", "abc", "1")
    assert upload_run.choose_run(_Publisher()) == "run-new"
    output = capsys.readouterr().out
    assert "stays local" in output and "Enter one of the numbers" in output


def test_picker_can_quit_and_survives_an_unreachable_shared_source(monkeypatch, runs):
    class Offline(_Publisher):
        def published_run_ids(self):
            raise ConnectionError("offline")

    _answers(monkeypatch, "q")
    assert upload_run.choose_run(Offline()) is None
