"""Deterministic checks that run in code rather than in the model.

A presence check ("was the recording notice given?") is decidable by looking for
the words, so it should not depend on the model noticing a line that ranked
fourth. `phrase_match` is the mechanical half: an exact, case-insensitive,
whole-word scan of the transcript that either finds the phrase or does not.

Used by retrieval to guarantee that a chunk literally containing one of the
item's terms is handed to the analysis node, whatever the lexical scores say.
"""
from __future__ import annotations

import re
from typing import Iterable

from ..state import Chunk


def phrase_match(text: str, terms: Iterable[str]) -> list[tuple[int, int]]:
    """Character offsets of every whole-word occurrence of any term, in order.

    Whole-word so that "fee" does not match "coffee". Terms may be multi-word
    phrases; internal whitespace matches any run of whitespace, so a phrase still
    matches across a line break.
    """
    hits: list[tuple[int, int]] = []
    for term in terms:
        term = (term or "").strip()
        if not term:
            continue
        pattern = r"\s+".join(re.escape(word) for word in term.split())
        for match in re.finditer(rf"(?<!\w){pattern}(?!\w)", text, re.IGNORECASE):
            hits.append((match.start(), match.end()))
    return sorted(set(hits))


def chunks_with_phrase(terms: Iterable[str], chunks: Iterable[Chunk]) -> list[Chunk]:
    """Every chunk containing at least one of the terms, in transcript order."""
    terms = list(terms)
    return [chunk for chunk in chunks if phrase_match(chunk["text"], terms)]
