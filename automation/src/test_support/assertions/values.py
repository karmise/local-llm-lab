"""Typed value and collection expectations with useful failure diagnostics."""

from collections.abc import Container, Iterable, Sized
from typing import Any


def equal(actual: object, expected: object, *, message: str | None = None) -> None:
    assert actual == expected, message or f"Expected {expected!r}, got {actual!r}"


def not_equal(actual: object, unexpected: object, *, message: str | None = None) -> None:
    assert actual != unexpected, message or f"Unexpected value: {actual!r}"


def identical(actual: object, expected: object, *, message: str | None = None) -> None:
    assert actual is expected, message or "Expected the original object to be preserved"


def not_identical(actual: object, unexpected: object, *, message: str | None = None) -> None:
    assert actual is not unexpected, message or "Expected distinct objects"


def truthy(actual: object, *, message: str | None = None) -> None:
    assert actual, message or f"Expected a truthy value, got {actual!r}"


def falsy(actual: object, *, message: str | None = None) -> None:
    assert not actual, message or f"Expected an empty or false value, got {actual!r}"


def length(actual: Sized, expected: int, *, message: str | None = None) -> None:
    equal(len(actual), expected, message=message)


def contains(container: Container, item: object, *, message: str | None = None) -> None:
    assert item in container, message or f"Expected {item!r} in {container!r}"


def excludes(container: Container, item: object, *, message: str | None = None) -> None:
    assert item not in container, message or f"Unexpected {item!r} in {container!r}"


def instance_of(
    actual: object, expected: type | tuple[type, ...], *, message: str | None = None
) -> None:
    assert isinstance(actual, expected), message or f"Expected {expected!r}, got {type(actual)!r}"


def all_true(values: Iterable[object], *, message: str | None = None) -> None:
    assert all(values), message or "Expected every checked condition to hold"


def less_than(actual: Any, expected: Any, *, message: str | None = None) -> None:
    assert actual < expected, message or f"Expected {actual!r} < {expected!r}"


def at_most(actual: Any, expected: Any, *, message: str | None = None) -> None:
    assert actual <= expected, message or f"Expected {actual!r} <= {expected!r}"


def greater_than(actual: Any, expected: Any, *, message: str | None = None) -> None:
    assert actual > expected, message or f"Expected {actual!r} > {expected!r}"


def at_least(actual: Any, expected: Any, *, message: str | None = None) -> None:
    assert actual >= expected, message or f"Expected {actual!r} >= {expected!r}"
