"""`context_lookup`: fetch client, product, and restricted-list records.

Final intended behaviour (CLAUDE.md): plain lookups against mock CSV tables
(later SQL) via a `MockRepository` class exposing `get_client(caller)`,
`get_products(ids)`, `get_restricted()`. Client key is `client_id` from the
input if given, else `client_context` stays `None`. Missing records are left
empty; checks that need them return `CANNOT_DETERMINE`.

Step-0 stub: always returns empty context.
"""
from __future__ import annotations

from compliance.state import ComplianceState


def context_lookup_node(state: ComplianceState) -> dict:
    return {"client_context": None, "product_context": []}
