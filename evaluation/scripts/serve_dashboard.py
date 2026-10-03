"""Start the evaluation dashboard: the /api endpoints plus the web app, on one port.

    python evaluation/scripts/serve_dashboard.py

The web app is built automatically the first time and whenever its source changed (this needs
Node.js; `npm install` also runs once). Restart the server after pulling backend changes.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EVALUATION_ROOT.parent
WEB_ROOT = EVALUATION_ROOT / "web"
sys.path.insert(0, str(EVALUATION_ROOT))

from dashboard.api import create_app  # noqa: E402


def needs_build(web: Path) -> bool:
    """True when web/dist is missing or older than any frontend source file."""
    index = web / "dist" / "index.html"
    if not index.is_file():
        return True
    built = index.stat().st_mtime
    inputs = [web / name for name in ("index.html", "package.json", "vite.config.ts", "tsconfig.json")]
    inputs += [path for path in (web / "src").rglob("*") if path.is_file()] if (web / "src").is_dir() else []
    return any(path.stat().st_mtime > built for path in inputs if path.is_file())


def build_frontend(web: Path) -> bool:
    """Install packages (first time) and build. Returns False, with a hint, if that is not possible."""
    npm = shutil.which("npm")
    if not npm:
        print("Node.js (npm) was not found, so the web app cannot be built. Install Node.js 20+ from "
              "https://nodejs.org, or serve only the API. ", file=sys.stderr)
        return False
    try:
        if not (web / "node_modules").is_dir():
            print("Installing web app packages (first time only) ...", flush=True)
            subprocess.run([npm, "install"], cwd=web, check=True)
        print("Building the web app ...", flush=True)
        subprocess.run([npm, "run", "build"], cwd=web, check=True)
    except (subprocess.CalledProcessError, OSError) as error:
        print(f"Web app build failed: {error}", file=sys.stderr)
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-source", choices=("shared", "local"),
                        help="Source opened at / (default: EVALUATION_DEFAULT_SOURCE or shared)")
    parser.add_argument("--no-build", action="store_true", help="Do not build the web app, even if it is missing or stale")
    args = parser.parse_args(argv)
    load_dotenv(REPOSITORY_ROOT / ".env")
    load_dotenv(EVALUATION_ROOT / ".env")
    if not args.no_build and needs_build(WEB_ROOT):
        build_frontend(WEB_ROOT)
    if not (WEB_ROOT / "dist" / "index.html").is_file():
        print("Note: the web app is not built, so only /api is served.")
    print(f"\nDashboard: http://{args.host}:{args.port}   (Ctrl+C to stop)\n", flush=True)
    uvicorn.run(create_app(default_source=args.default_source), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
