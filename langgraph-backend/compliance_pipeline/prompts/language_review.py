"""NLP-only language and transaction-scope guidance; no dataset dependencies."""

LANGUAGE_REVIEW = """
Review English, Singlish, Mandarin and code-switching by meaning, using the same
requirements. Dialect, particles and informal phrasing are not themselves breaches
or reasons to abstain. Evidence quotes must copy the ORIGINAL-language segment;
translate only in your explanation, never inside a quote. Preserve names, numbers,
currencies, negation and speaker attribution. Do not rewrite Chinese numerals or
financial shorthand in quoted evidence.
Include any qualifying negation or condition rather than quoting a fragment that
reverses the speaker's meaning.

Track corrections and withdrawals across the entire conversation. For facts about
the final instruction, use the client's latest unwithdrawn instruction; an advisor's
incorrect read-back does not change it without client acceptance. A complete order
replacement supersedes the previous order; an unrelated product or historic amount
is not the final instruction. Do not erase valid earlier disclosures merely because
a different detail was corrected later.

Brief or dialectal authorisation can qualify when its context clearly authorises
the final instruction. A filler, politeness, acknowledgement or explanation of
motives alone is insufficient. Distinguish refusal from consent, a request for a
guarantee from an employee making one, and willingness to accept loss from evidence
of understanding the particular risk. Do not infer a missing explanation from
separate nearby facts. Missing timestamps do not establish missing conversation;
judge completeness from the supplied text and its context.
"""
