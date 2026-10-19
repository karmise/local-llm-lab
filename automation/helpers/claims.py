"""Split an answer into the sentences that are checked as claims."""

import re

# A sentence ends at . ! or ?, optionally followed by a closing quote or bracket, before a capitalised start.
_SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')]))\s+(?=[A-Z\"'(])")


def split_claims(text: str) -> list[str]:
    """Sentences of ``text`` that contain a word; a claim is one sentence of the answer, unchanged."""
    return [part.strip() for part in _SENTENCE_END.split(text.strip()) if re.search(r"[A-Za-z]", part)]
