"""Persist a performance experiment before enforcing its declared acceptance gate."""

import platform
from pathlib import Path
from typing import Any
from uuid import uuid4

from llm_testkit import assertions
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.performance.runner import validate_batch
from llm_testkit.reporting.steps import attach_file


def record_batch(
        report: dict[str, Any], directory: Path, *, base_url: str, timeout: float, maximum_p95: float,
        metadata: dict[str, Any]) -> Path:
    validate_batch(report)
    evidence = {
            **report, "metadata": {
            **metadata, "python": platform.python_version(),
            "system": platform.platform(),
            "machine": platform.node(),
            "base_url": base_url,
            "timeout": timeout,
            "warmup_requests": 0},
            "thresholds": {
            "maximum_p95": maximum_p95,
            "maximum_failure_rate": 0.0}}
    path = directory / f"{uuid4().hex}.json"
    write_sample(path, evidence)
    attach_file(path, name="Performance attempts and thresholds", media_type="application/json", extension="json")
    assertions.assert_performance_batch(evidence, maximum_p95=maximum_p95)
    return path
