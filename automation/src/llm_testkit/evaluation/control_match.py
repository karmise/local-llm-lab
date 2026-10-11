"""Compare a judge's control result with its hand labels without relying on assert statements."""

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from llm_testkit.core.number_words import normalize_number_words


def _score_problem(value: Any) -> str | None:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        return f"Expected a finite quality score between 0 and 1, got {value!r}"
    return None


def calibration_mismatch(
        result: Mapping[str, Any], *, expected_score: float, claims: Sequence[Mapping[str, Any]]) -> str | None:
    """Why ``result`` differs from the labelled control, or None when it matches.

    Unlike an ``assert``, this check also runs under ``python -O``.
    """
    score = result["value"]
    problem = _score_problem(score) or _score_problem(expected_score)
    if problem:
        return problem
    if not math.isclose(score, expected_score, rel_tol=0, abs_tol=1e-9):
        return f"Control score: expected {expected_score}, got {score}"
    if "verdicts" not in result:
        return "Missing required field: verdicts"
    verdicts = result["verdicts"]
    if type(verdicts) is not list:
        return f"Field verdicts: expected list, got {type(verdicts).__name__}"
    if len(verdicts) != len(claims):
        return "Control extraction changed the expected number of claims"
    matched_indices: set[int] = set()
    for claim in claims:
        matches = [
                index for index, item in enumerate(verdicts)
                if re.search(claim["pattern"], item["statement"], flags=re.IGNORECASE)
                or re.search(claim["pattern"], normalize_number_words(item["statement"]), flags=re.IGNORECASE)]
        if len(matches) != 1:
            return f"Expected one extracted claim matching {claim['pattern']}"
        index = matches[0]
        if index in matched_indices:
            return "Control claims must map to distinct extracted statements"
        matched_indices.add(index)
        verdict = verdicts[index]
        if "verdict" not in verdict:
            return "Missing required field: verdict"
        actual = verdict["verdict"]
        if type(actual) is not type(claim["verdict"]):
            return f"Field verdict: expected {type(claim['verdict']).__name__}, got {type(actual).__name__}"
        if actual != claim["verdict"]:
            return f"Field verdict: expected {claim['verdict']!r}, got {actual!r}"
    return None
