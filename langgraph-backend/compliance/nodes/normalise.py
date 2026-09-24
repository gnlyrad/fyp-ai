"""`normalise`: turn extracted orders into clean, matched data (code, chained after `extraction`).

Final intended behaviour (CLAUDE.md / plan): parse amounts ("5k" -> 5000,
"1.2m" -> 1200000, basic number words), map currency symbols/words to ISO
4217 codes, and fuzzy-match `product_name` against `products.csv`
(`rapidfuzz`, score >= 85) to fill in `product_id`.

Step-0 stub: no-op.
"""
from __future__ import annotations

from compliance.state import ComplianceState


def normalise_node(state: ComplianceState) -> dict:
    return {}
