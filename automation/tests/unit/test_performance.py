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
        "machine": "test-machine",
        "python": "3.12",
        "base_url": "http://localhost",
        "warmup_requests": 0,
        "timeout": 5,
    }
    current = deepcopy(base)
    if change == "slower":
        current["attempts"][0]["elapsed_seconds"] *= 2
        current["wall_seconds"] *= 2
        current["completed_requests_per_second"] /= 2
        value = current["attempts"][0]["elapsed_seconds"]
        current["latency_seconds"] = dict.fromkeys(("minimum", "median", "p95", "maximum"), value)
    elif change == "machine":
        current["metadata"]["system"] = "other"
    elif change == "missing":
        current.pop("metadata")
    assert compare_batches(base, current)["status"] == expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("users", True),
        ("users", 5),
        ("requests", True),
        ("schema_version", True),
        ("wall_seconds", True),
    ],
)
def test_saved_batch_revalidates_budget_and_numeric_types(field, value):
    report = run_batch(lambda: None)
    report[field] = value
    with pytest.raises(ValueError):
        validate_batch(report)


@pytest.mark.parametrize(
    "field",
    [
        "timeout",
        "policy_sha256",
        "golden_dataset_sha256",
        "case_id",
        "model_digest",
        "configuration",
    ],
)
def test_rag_comparison_requires_explicit_provenance(field):
    from llm_testkit.performance.comparison import compare_batches

    base = run_batch(lambda: None)
    base["metadata"] = {
        "workload": "rag",
        "system": "test",
        "machine": "test-machine",
        "python": "3.12",
        "base_url": "http://localhost",
        "warmup_requests": 0,
        "timeout": 60,
        "policy_sha256": "a" * 64,
        "golden_dataset_sha256": "b" * 64,
        "case_id": "carryover_limit",
        "model_digest": "c" * 64,
        "generation_model": "model",
        "configuration": {"chatModel": "model", "topN": 4},
    }
    base["metadata"].pop(field)
    assert compare_batches(base, deepcopy(base))["status"] == "incomparable"


def test_incompatible_capture_mode_fails_before_external_setup(framework_pytester):
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    framework_pytester.makepyfile("""
        import pytest
        @pytest.mark.performance
        @pytest.mark.rag
        def test_batch(missing_external_service): pass
    """)
    result = framework_pytester.runpytest_subprocess(
        "--run-performance", "--performance-mode=rag", "--capture-rag", "-q"
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR


@pytest.mark.parametrize("change", ["timeout", "machine", "empty-machine", "boolean-warmup"])
def test_health_comparison_requires_matching_execution_conditions(change):
    from llm_testkit.performance.comparison import compare_batches

    base = run_batch(lambda: None)
    base["metadata"] = {
        "workload": "health",
        "system": "test",
        "machine": "first",
        "python": "3.12",
        "base_url": "http://localhost",
        "warmup_requests": 0,
        "timeout": 5,
    }
    current = deepcopy(base)
    if change == "empty-machine":
        base["metadata"]["machine"] = current["metadata"]["machine"] = ""
    elif change == "boolean-warmup":
        current["metadata"]["warmup_requests"] = False
    else:
        current["metadata"][change] = 10 if change == "timeout" else "second"
    assert compare_batches(base, current)["status"] == "incomparable"


@pytest.mark.parametrize("failure", [False, True])
def test_recorded_batches_retain_actual_timeout_and_failures(tmp_path, failure):
    import json

    from llm_testkit.performance.reporting import record_batch

    def operation():
        if failure:
            raise TimeoutError("retained")

    report = run_batch(operation)
    kwargs = dict(
        base_url="http://localhost", timeout=3, maximum_p95=10, metadata={"workload": "health"}
    )
    if failure:
        with pytest.raises(AssertionError, match="failure rate"):
            record_batch(report, tmp_path, **kwargs)
    else:
        record_batch(report, tmp_path, **kwargs)
    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1
    saved = json.loads(paths[0].read_text())
    assert saved["metadata"]["timeout"] == 3
    assert saved["metadata"]["machine"]
    assert "metadata" not in report
    assert saved["failed"] == int(failure)
    if failure:
        assert saved["attempts"][0]["error"] == "retained"


@pytest.mark.parametrize("change", ["row", "index", "status", "elapsed", "wall", "failed", "rps"])
def test_malformed_saved_attempts_are_rejected_as_validation_errors(change):
    report = run_batch(lambda: None)
    if change == "row":
        report["attempts"][0] = None
    elif change == "index":
        report["attempts"][0]["index"] = False
    elif change == "status":
        report["attempts"][0].pop("status")
    elif change == "elapsed":
        report["attempts"][0]["elapsed_seconds"] = "0"
    elif change == "wall":
        report["wall_seconds"] = report["attempts"][0]["elapsed_seconds"] / 2
    elif change == "failed":
        report["failed"] = False
    else:
        report["completed_requests_per_second"] = True
    with pytest.raises(ValueError):
        validate_batch(report)
