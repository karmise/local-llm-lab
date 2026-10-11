"""Quality measurements: scores, judge controls, report dimensions, benchmarks and performance batches."""

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from llm_testkit.evaluation.control_match import calibration_mismatch
from llm_testkit.reporting.steps import step

if TYPE_CHECKING:
    pass
from llm_testkit.assertions.fields import assert_field_type


def assert_quality_score(value: float) -> None:
    assert type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1, (
            f"Expected a finite quality score between 0 and 1, got {value!r}")


def assert_calibration_result(
        result: Mapping[str, Any], *, expected_score: float, claims: Sequence[Mapping[str, Any]]) -> None:
    mismatch = calibration_mismatch(result, expected_score=expected_score, claims=claims)
    assert mismatch is None, mismatch


def assert_benchmark_report(report: Mapping[str, Any]) -> None:
    assert report.get("schema_version") == 1, "Unsupported benchmark schema"
    assert report.get("status") == "checks_passed", (
            f"Benchmark did not pass: {report.get('status')}; "
            f"summary={report.get('summary')}; error={report.get('error')}")


def assert_benchmark_case(row: Mapping[str, Any]) -> None:
    assert not row.get("error"), f"Benchmark case failed: {row.get('error')}"
    assert row.get("generation_status") == "passed", (
            f"Generation/setup/teardown did not pass: {row.get('generation_status')}")
    for dimension in row["dimensions"]:
        if dimension["status"] != "not_applicable":
            assert_quality_dimension(dimension)


def assert_quality_dimension(dimension: Mapping[str, Any]) -> None:
    assert dimension.get("status") in ("passed", "measured"), (
            f"{dimension.get('name', 'Quality dimension')}: {dimension.get('error', dimension.get('status'))}")


def assert_quality_report(report: Mapping[str, Any]) -> None:
    dimensions = assert_field_type(report, "dimensions", list)
    assert 3 <= len(dimensions) <= 6, "Expected base dimensions with optional correctness/relevance"
    for dimension in dimensions:
        assert_quality_dimension(dimension)


@step("Check: performance batch meets explicit latency and failure thresholds")
def assert_performance_batch(
        report: Mapping[str, Any], *, maximum_p95: float, maximum_failure_rate: float = 0.0) -> None:
    from llm_testkit.performance.runner import validate_batch

    validate_batch(dict(report))
    assert type(maximum_p95) in (int,
            float) and math.isfinite(maximum_p95) and maximum_p95 > 0, ("Invalid p95 threshold")
    assert_quality_score(maximum_failure_rate)
    assert report["failed"] / report["requests"] <= maximum_failure_rate, ("Performance failure rate exceeds threshold")
    assert report["latency_seconds"]["p95"] <= maximum_p95, "Performance p95 exceeds threshold"
