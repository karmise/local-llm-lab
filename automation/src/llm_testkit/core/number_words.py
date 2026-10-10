"""Canonicalize unambiguous English integer spellings for control claim matching."""

import re

_SMALL_WORDS = "zero one two three four five six seven eight nine ten eleven"
_SMALL_WORDS += " twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
SMALL = dict(zip(_SMALL_WORDS.split(), range(20), strict=True))
TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10), strict=True))
NUMBERS = SMALL | TENS
WORDS = "|".join((*SMALL, *TENS, "hundred", "thousand", "million", "billion"))
PHRASE = re.compile(rf"\b(?:{WORDS})(?:[- ]+(?:(?:and|point)[- ]+)?(?:{WORDS}))*\b", re.IGNORECASE)


def normalize_number_words(text: str) -> str:
    """Normalize integers 0–99; preserve larger, decimal or ambiguous phrases."""
    def replace(match: re.Match) -> str:
        parts = re.split(r"[- ]+", match.group().lower())
        if len(parts) == 1 and parts[0] in NUMBERS:
            return str(NUMBERS[parts[0]])
        if len(parts) == 2 and parts[0] in TENS and parts[1] in SMALL and 0 < SMALL[parts[1]] < 10:
            return str(TENS[parts[0]] + SMALL[parts[1]])
        return match.group()

    return PHRASE.sub(replace, text)
