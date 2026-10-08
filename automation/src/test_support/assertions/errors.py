"""Exception expectations preserve pytest's type, regex and captured-error behavior."""

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TypeVar

import pytest

E = TypeVar("E", bound=BaseException)


def expected_error(expected: type[E] | tuple[type[E], ...], *,
        match: str | None = None) -> AbstractContextManager[pytest.ExceptionInfo[E]]:
    return pytest.raises(expected, match=match)


def rejects(operation: Callable[[], object], *, expected: type[E] | tuple[type[E], ...],
        match: str | None = None) -> pytest.ExceptionInfo[E]:
    with expected_error(expected, match=match) as caught:
        operation()
    return caught
