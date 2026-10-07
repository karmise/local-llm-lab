"""Expected outcomes of isolated child pytest runs."""

from typing import Any


def outcomes(result: Any, **expected: int) -> None:
    result.assert_outcomes(**expected)
