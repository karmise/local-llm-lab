import json
from copy import deepcopy
from threading import Lock

import pytest

from llm_testkit import assertions
from llm_testkit.performance.comparison import compare_batches
from llm_testkit.performance.runner import run_batch, validate_batch
from llm_testkit.reporting.steps import title
from test_support.builders.performance import (
    check_recorded_batches_retain_actual_timeout_and_failures_outcome,
    check_recorded_batches_retain_actual_timeout_and_failures_outcome_2,
    make_operation_stub,
    make_operation_stub_2,
    prepare_evidence_case,
    prepare_health_comparison_requires_matching_execution_conditions_case,
    prepare_malformed_saved_attempts_are_rejected_as_validation_errors_case,
    prepare_performance_comparison_case,
)
from test_support.data.performance import (
    BUDGET_REQUESTS_USERS_CASES,
    EVIDENCE_CHANGE_CASES,
    HEALTH_COMPARISON_REQUIRES_MATCHING_EXECUTION_CONDITIONS_CHANGE_CASES,
    MALFORMED_SAVED_ATTEMPTS_ARE_REJECTED_AS_VALIDATION_ERRORS_CHANGE_CASES,
    PERFORMANCE_COMPARISON_CHANGE_EXPECTED_CASES,
    RAG_COMPARISON_REQUIRES_EXPLICIT_PROVENANCE_FIELD_CASES,
    RECORDED_BATCHES_RETAIN_ACTUAL_TIMEOUT_AND_FAILURES_FAILURE_CASES,
    SAVED_BATCH_REVALIDATES_BUDGET_AND_NUMERIC_TYPES_FIELD_VALUE_CASES,
)
from test_support.data.scripts.performance import (
    INCOMPATIBLE_CAPTURE_MODE_FAILS_BEFORE_EXTERNAL_SETUP_MAKEPYFILE_SOURCE,
    SELECTION_MAKEPYFILE_SOURCE,
)

pytestmark = pytest.mark.unit


@title("Concurrent batches execute the exact request budget and retain errors without retries")
def test_bounded_batch():
    lock = Lock()
    calls = []

    operation = make_operation_stub(calls, lock)

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


@pytest.mark.parametrize(("requests", "users"), BUDGET_REQUESTS_USERS_CASES)
@title("Invalid load budgets fail before making requests [{param_id}]")
def test_budget(requests, users):
    calls = []
    with pytest.raises(ValueError):
        run_batch(lambda: calls.append(1), requests=requests, users=users)
    assert not calls


@pytest.mark.parametrize("change", EVIDENCE_CHANGE_CASES)
@title(
    "Performance summaries reject missing attempts and inconsistent or nonfinite measurements [{param_id}]"
)
def test_evidence(change):
    report = run_batch(lambda: None, requests=2)
    altered = deepcopy(report)
    prepare_evidence_case(altered, change)
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
    framework_pytester.makepyfile(SELECTION_MAKEPYFILE_SOURCE)
    framework_pytester.runpytest_subprocess("-q").assert_outcomes(skipped=2)
    result = framework_pytester.runpytest_subprocess(
        "--run-performance", "--performance-requests", "21", "-q"
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR


@pytest.mark.parametrize(
    ("change", "expected"),
    PERFORMANCE_COMPARISON_CHANGE_EXPECTED_CASES,
)
@title(
    "Performance baselines detect slowdown while rejecting changed or missing experiment metadata [{param_id}]"
)
def test_performance_comparison(change, expected):

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
    prepare_performance_comparison_case(change, current)
    assert compare_batches(base, current)["status"] == expected


@pytest.mark.parametrize(
    "field,value",
    SAVED_BATCH_REVALIDATES_BUDGET_AND_NUMERIC_TYPES_FIELD_VALUE_CASES,
)
def test_saved_batch_revalidates_budget_and_numeric_types(field, value):
    report = run_batch(lambda: None)
    report[field] = value
    with pytest.raises(ValueError):
        validate_batch(report)


@pytest.mark.parametrize(
    "field",
    RAG_COMPARISON_REQUIRES_EXPLICIT_PROVENANCE_FIELD_CASES,
)
def test_rag_comparison_requires_explicit_provenance(field):

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
    framework_pytester.makepyfile(
        INCOMPATIBLE_CAPTURE_MODE_FAILS_BEFORE_EXTERNAL_SETUP_MAKEPYFILE_SOURCE
    )
    result = framework_pytester.runpytest_subprocess(
        "--run-performance", "--performance-mode=rag", "--capture-rag", "-q"
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR


@pytest.mark.parametrize(
    "change", HEALTH_COMPARISON_REQUIRES_MATCHING_EXECUTION_CONDITIONS_CHANGE_CASES
)
def test_health_comparison_requires_matching_execution_conditions(change):

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
    prepare_health_comparison_requires_matching_execution_conditions_case(base, change, current)
    assert compare_batches(base, current)["status"] == "incomparable"


@pytest.mark.parametrize(
    "failure", RECORDED_BATCHES_RETAIN_ACTUAL_TIMEOUT_AND_FAILURES_FAILURE_CASES
)
def test_recorded_batches_retain_actual_timeout_and_failures(tmp_path, failure):

    operation = make_operation_stub_2(failure)

    report = run_batch(operation)
    kwargs = dict(
        base_url="http://localhost", timeout=3, maximum_p95=10, metadata={"workload": "health"}
    )
    check_recorded_batches_retain_actual_timeout_and_failures_outcome(
        failure, kwargs, report, tmp_path
    )
    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1
    saved = json.loads(paths[0].read_text())
    assert saved["metadata"]["timeout"] == 3
    assert saved["metadata"]["machine"]
    assert "metadata" not in report
    assert saved["failed"] == int(failure)
    check_recorded_batches_retain_actual_timeout_and_failures_outcome_2(failure, saved)


@pytest.mark.parametrize(
    "change", MALFORMED_SAVED_ATTEMPTS_ARE_REJECTED_AS_VALIDATION_ERRORS_CHANGE_CASES
)
def test_malformed_saved_attempts_are_rejected_as_validation_errors(change):
    report = run_batch(lambda: None)
    prepare_malformed_saved_attempts_are_rejected_as_validation_errors_case(change, report)
    with pytest.raises(ValueError):
        validate_batch(report)
