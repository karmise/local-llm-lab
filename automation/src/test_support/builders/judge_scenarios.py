"""Ready-to-use judge scenarios; construction is owned by scoped fixtures."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock


@dataclass
class JudgeScenario:
    evaluate: Callable[[], Any]
    judge: Any
    client: Mock
    expected_score: float
    maximum_calls: int
    check_options: bool = False


@dataclass
class RelevanceScenario:
    evaluate: Callable[[], Any]
    precision: Any
    recall: Any
    expected: dict[str, Any]


@dataclass
class FailedServiceScenario:
    evaluate: Callable[[], Any]
    transport: Mock
    judge: Mock | None = None
    precision: Mock | None = None
    recall: Mock | None = None


def run_async(operation: Callable[[], Any]) -> Any:
    return asyncio.run(operation())
