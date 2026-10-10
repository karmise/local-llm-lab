"""Canonicalize unambiguous English integer spellings for control claim matching."""

import re

_SMALL_WORDS = "zero one two three four five six seven eight nine ten eleven"
_SMALL_WORDS += " twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
SMALL = dict(zip(_SMALL_WORDS.split(), range(20), strict=True))
TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10), strict=True))
SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
NUMBERS = SMALL | TENS
WORDS = "|".join((*SMALL, *TENS, "hundred", *SCALES))
PHRASE = re.compile(rf"\b(?:{WORDS})(?:[- ]+(?:(?:and|point)[- ]+)?(?:{WORDS}))*\b", re.IGNORECASE)


def _integer(words: list[str]) -> int | None:
    """The integer spelled by ``words`` (for example "five thousand two hundred and ten"), or None if the
    spelling is not one well-formed integer."""
    total = group = 0
    previous = None
    last_scale = None
    for word in words:
        if word == "and":
            if previous not in ("hundred", "scale"):
                return None
            continue
        if word in SMALL:
            if previous == "small" or (previous == "tens" and not 0 < SMALL[word] < 10):
                return None
            if word == "zero" and len(words) > 1:
                return None
            group += SMALL[word]
            previous = "small"
        elif word in TENS:
            if previous in ("small", "tens") or group % 100:
                return None
            group += TENS[word]
            previous = "tens"
        elif word == "hundred":
            if previous != "small" or not 0 < group < 10:
                return None
            group *= 100
            previous = "hundred"
        else:
            scale = SCALES[word]
            if previous is None or previous == "scale" or group == 0 or (last_scale and scale >= last_scale):
                return None
            total += group * scale
            group, last_scale, previous = 0, scale, "scale"
    return None if previous is None else total + group


def normalize_number_words(text: str) -> str:
    """Write well-formed integer spellings as digits; preserve decimals and malformed phrases."""
    def replace(match: re.Match) -> str:
        words = re.split(r"[- ]+", match.group().lower())
        if "point" in words:
            return match.group()
        value = _integer(words)
        return match.group() if value is None else str(value)

    return PHRASE.sub(replace, text)
