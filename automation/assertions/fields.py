"""Response status, JSON body and typed field checks shared by every other check."""

from collections.abc import Mapping, Sized
from typing import TYPE_CHECKING, Any, TypeVar

from requests import Response

if TYPE_CHECKING:
    pass

T = TypeVar("T")


def assert_status_code(response: Response, expected: int, *, context: str = "Response") -> None:
    assert response.status_code == expected, (f"{context}: expected HTTP {expected}, got {response.status_code}")


def assert_json_object(response: Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        raise AssertionError("Response body is not valid JSON") from None
    assert isinstance(payload, dict), "Expected a JSON object in the response body"
    return payload


def assert_field_type(payload: Mapping[str, Any], field: str, expected: type[T]) -> T:
    assert field in payload, f"Missing required field: {field}"
    value = payload[field]
    assert type(value) is expected, (f"Field {field}: expected {expected.__name__}, got {type(value).__name__}")
    return value


def assert_field_equals(payload: Mapping[str, Any], field: str, expected: Any) -> None:
    value = assert_field_type(payload, field, type(expected))
    assert value == expected, f"Field {field}: expected {expected!r}, got {value!r}"


def assert_field_length(payload: Mapping[str, Any], field: str, expected: int) -> None:
    assert field in payload, f"Missing required field: {field}"
    value = payload[field]
    assert isinstance(value, Sized), f"Field {field} does not have a length"
    assert len(value) == expected, f"Field {field}: expected length {expected}, got {len(value)}"


def assert_field_contains(payload: Mapping[str, Any], field: str, expected: str) -> None:
    value = assert_field_type(payload, field, str)
    assert expected in value, f"Field {field}: expected to contain {expected!r}"


def assert_field_starts_with(payload: Mapping[str, Any], field: str, expected: str) -> None:
    value = assert_field_type(payload, field, str)
    assert value.startswith(expected), f"Field {field}: expected prefix {expected!r}"
