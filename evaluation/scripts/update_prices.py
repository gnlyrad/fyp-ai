"""Refresh evaluation/model_pricing.json from the public price catalog.

    python -B evaluation/scripts/update_prices.py                  # models already in the file + the pipeline's model
    python -B evaluation/scripts/update_prices.py --model gpt-x    # add or refresh specific models

Runs price themselves automatically from the catalog, so this is optional. Use it to keep a
visible price book, to see when a price changed (the previous price is kept in `history`),
or to pin prices before an offline run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

EVALUATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVALUATION_ROOT))

from dotenv import load_dotenv  # noqa: E402

from fyp_evaluation_engine.usage import (  # noqa: E402
    CATALOG_NAME,
    PRICING_FILE,
    catalog_price,
    catalog_url,
    fetch_catalog,
    load_pricing,
    model_key,
    refresh_entry,
)


def _line(model: str, entry: dict) -> str:
    cached = f", cached {entry['cached_input']}" if entry.get("cached_input") is not None else ""
    return f"{model}: input {entry['input']}, output {entry['output']}{cached} per 1M tokens"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", default=[], help="Model id to add or refresh (repeatable)")
    parser.add_argument("--yes", action="store_true", help="Write without asking")
    args = parser.parse_args(argv)
    load_dotenv(EVALUATION_ROOT.parent / ".env")

    path = EVALUATION_ROOT / PRICING_FILE
    pricing = load_pricing(path)
    models = [model_key(model) for model in args.model] or list(pricing.get("models") or {})
    if not models:
        models = [model_key(os.getenv("COMPLIANCE_MODEL_ID") or os.getenv("CLEARCALL_MODEL_ID") or "gemini-3.5-flash")]
    if not catalog_url():
        print("Price lookup is switched off (EVALUATION_PRICE_CATALOG_URL).", file=sys.stderr)
        return 2
    print(f"Fetching {CATALOG_NAME} ...")
    catalog = fetch_catalog(timeout=30)
    if not catalog:
        print("Could not download the price catalog. Check your connection; nothing was changed.", file=sys.stderr)
        return 2

    updated = {**pricing, "models": dict(pricing.get("models") or {})}
    changed = False
    for model in models:
        found = catalog_price(catalog, model)
        if not found:
            print(f"  {model}: not in the catalog, left as is (add a price by hand if you need one)")
            continue
        entry, change = refresh_entry(updated["models"].get(model), found)
        updated["models"][model] = entry
        if change == "unchanged":
            print(f"  {_line(model, entry)}  (unchanged)")
        else:
            old = (entry["history"] or [None])[-1] if change == "changed" else None
            print(f"  {_line(model, entry)}  ({'NEW' if change == 'new' else 'CHANGED'})")
            if old:
                print(f"    was: input {old['input']}, output {old['output']} (as of {old.get('as_of')})")
            changed = True
    if not changed:
        print("Prices are up to date.")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("Run in a terminal to confirm, or pass --yes.", file=sys.stderr)
            return 2
        if input("\nWrite these prices to model_pricing.json? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing written.")
            return 0
    path.write_text(json.dumps(updated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Updated {path.name}. Past runs keep the prices they were made with.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
