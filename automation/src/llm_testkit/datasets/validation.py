"""Shared validation for reviewed JSON catalogs; never coerce malformed input."""

import json
import re
from typing import Any


def require_object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def load_catalog(raw: bytes, context: str) -> dict[str, Any]:
    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{context}: duplicate JSON field: {key}")
            result[key] = value
        return result

    data = require_object(json.loads(raw, object_pairs_hook=unique_fields), context)
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError(f"Unsupported {context} schema; expected integer schema_version 1")
    return data


def require_text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a nonempty string")
    return value


def validate_patterns(value: Any, context: str, *, required: bool) -> tuple[tuple[str, str], ...]:
    patterns = require_object(value, context)
    if required and not patterns:
        raise ValueError(f"{context} must be a nonempty object")
    for label, pattern in patterns.items():
        require_text(label, f"{context} label")
        require_text(pattern, f"{context}.{label}")
        try:
            compiled = re.compile(pattern, flags=re.IGNORECASE)
        except re.error as error:
            raise ValueError(f"{context}.{label}: invalid regex: {error}") from error
        if compiled.search(""):
            raise ValueError(f"{context}.{label}: regex must not match empty text")
    return tuple(patterns.items())
