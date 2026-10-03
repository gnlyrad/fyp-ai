from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_local_dashboard_opens_dataset_and_run_history(monkeypatch):
    monkeypatch.setenv("EVALUATION_DEFAULT_SOURCE", "Local")
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    assert not app.exception
    assert app.title[0].value == "Evaluation datasets"
    assert app.selectbox[0].value.name == "pilot"
    app.button[0].click().run()
    assert not app.exception
    assert app.title[0].value == "Pilot"
    assert any(item.value == "Run history" for item in app.subheader)
