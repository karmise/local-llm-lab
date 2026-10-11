"""Validate quality scores with exceptions that survive python -O."""

import math
from typing import Any


def require_score(value: Any, name: str = "Quality score") -> float:
    """Return ``value`` if it is a finite number from 0 to 1; raise ValueError naming ``name`` otherwise."""
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number between 0 and 1, got {value!r}")
    return value
