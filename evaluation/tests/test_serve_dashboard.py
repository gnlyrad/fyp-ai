"""serve_dashboard.py rebuilds the web app only when it must."""
import importlib.util
import os
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serve_dashboard.py"
spec = importlib.util.spec_from_file_location("serve_dashboard_script", SCRIPT)
serve = importlib.util.module_from_spec(spec)
sys.modules["serve_dashboard_script"] = serve
spec.loader.exec_module(serve)


def _site(tmp_path):
    web = tmp_path / "web"
    (web / "src").mkdir(parents=True)
    (web / "dist").mkdir()
    for name in ("index.html", "package.json", "src/main.tsx"):
        (web / name).write_text("x", encoding="utf-8")
    (web / "dist" / "index.html").write_text("built", encoding="utf-8")
    return web


def _age(path, seconds):
    stamp = path.stat().st_mtime + seconds
    os.utime(path, (stamp, stamp))


def test_missing_build_is_built(tmp_path):
    web = _site(tmp_path)
    (web / "dist" / "index.html").unlink()
    assert serve.needs_build(web)


def test_fresh_build_is_left_alone(tmp_path):
    web = _site(tmp_path)
    _age(web / "dist" / "index.html", 100)  # built after every source file
    assert not serve.needs_build(web)


def test_a_newer_source_file_triggers_a_rebuild(tmp_path):
    web = _site(tmp_path)
    _age(web / "dist" / "index.html", 100)
    _age(web / "src" / "main.tsx", 200)  # edited after the build
    assert serve.needs_build(web)
