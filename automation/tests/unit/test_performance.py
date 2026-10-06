from copy import deepcopy
from threading import Lock

import pytest

from llm_testkit import assertions
from llm_testkit.performance.runner import run_batch, validate_batch
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit


@title("Concurrent batches execute the exact request budget and retain errors without retries")
def test_bounded_batch():
    lock = Lock()
    calls = []

    def operation():
        with lock:
            calls.append(1)
            index = len(calls)
        if index == 2:
            raise TimeoutError("Timeout")

    report = run_batch(operation, requests=4, users=2)
    assert len(calls) == 4
    assert report["failed"] == 1
    assert (
        next(r for r in report["attempts"] if r["status"] == "failed")["error_type"]
        == "TimeoutError"
    )
    validate_batch(report)
    with pytest.raises(AssertionError, match="failure rate"):
        assertions.assert_performance_batch(report, maximum_p95=10)
    assertions.assert_performance_batch(report, maximum_p95=10, maximum_failure_rate=0.25)


@pytest.mark.parametrize(("requests", "users"), [(0, 1), (21, 1), (2, 3), (5, 5), (True, 1)])
@title("Invalid load budgets fail before making requests [{param_id}]")
def test_budget(requests, users):
    calls = []
    with pytest.raises(ValueError):
        run_batch(lambda: calls.append(1), requests=requests, users=users)
    assert not calls


@pytest.mark.parametrize("change", ["attempts", "latency", "rps", "failed", "nan"])
@title(
    "Performance summaries reject missing attempts and inconsistent or nonfinite measurements [{param_id}]"
)
def test_evidence(change):
    report = run_batch(lambda: None, requests=2)
    altered = deepcopy(report)
    if change == "attempts":
        altered["attempts"].pop()
    elif change == "latency":
        altered["latency_seconds"]["p95"] = 99
    elif change == "rps":
        altered["completed_requests_per_second"] = 0
    elif change == "failed":
        altered["failed"] = 2
    else:
        altered["attempts"][0]["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError):
        validate_batch(altered)


@title("Latency gate accepts equality and rejects a lower declared threshold")
def test_latency_gate():
    report = run_batch(lambda: None)
    value = report["latency_seconds"]["p95"]
    assertions.assert_performance_batch(report, maximum_p95=value)
    with pytest.raises(AssertionError, match="p95 exceeds"):
        assertions.assert_performance_batch(report, maximum_p95=value / 2)


@title(
    "Performance scenarios are opt-in and reject oversized selected batches before fixture setup"
)
def test_selection(framework_pytester):
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    framework_pytester.makepyfile("""
        import pytest
        @pytest.mark.performance
        def test_health(missing_api): pass
        @pytest.mark.performance
        @pytest.mark.rag
        def test_rag(missing_model): pass
    """)
    framework_pytester.runpytest_subprocess("-q").assert_outcomes(skipped=2)
    result = framework_pytester.runpytest_subprocess(
        "--run-performance", "--performance-requests", "21", "-q"
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("none", "passed"),
        ("slower", "regression"),
        ("machine", "incomparable"),
        ("missing", "incomparable"),
    ],
)
@title(
    "Performance baselines detect slowdown while rejecting changed or missing experiment metadata [{param_id}]"
)
def test_performance_comparison(change, expected):
    from llm_testkit.performance.comparison import compare_batches

    base = run_batch(lambda: None)
    base["metadata"] = {
        "workload": "health",
        "system": "test",
        "python": "3.12",
        "base_url": "http://localhost",
        "warmup_requests": 0,
    }
    current = deepcopy(base)
    if change == "slower":
        current["attempts"][0]["elapsed_seconds"] *= 2
        value = current["attempts"][0]["elapsed_seconds"]
        current["latency_seconds"] = dict.fromkeys(("minimum", "median", "p95", "maximum"), value)
    elif change == "machine":
        current["metadata"]["system"] = "other"
    elif change == "missing":
        current.pop("metadata")
    assert compare_batches(base, current)["status"] == expected
