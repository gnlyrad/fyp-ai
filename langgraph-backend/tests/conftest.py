"""Keep production unit tests independent of a developer's enabled batch profile."""
import pytest


@pytest.fixture(autouse=True)
def baseline_question_batch_size(monkeypatch):
    # Tests of grouped requests select their own batch size explicitly.
    monkeypatch.setenv("COMPLIANCE_CHECK_BATCH_SIZE", "1")
    monkeypatch.setenv("COMPLIANCE_CHECK_CACHE_WARMUP", "0")
    monkeypatch.setenv("COMPLIANCE_CHECK_TRANSPORT", "standard")
    monkeypatch.setenv("COMPLIANCE_CHECK_RECOVERY_BATCH_SIZE", "1")
