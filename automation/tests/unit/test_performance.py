import json
from copy import deepcopy

import pytest

from llm_testkit import assertions
from llm_testkit.performance.comparison import compare_batches
from llm_testkit.performance.reporting import record_batch
from llm_testkit.performance.runner import run_batch, validate_batch
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import pytest_runs
from test_support.assertions import values as value_checks
from test_support.builders.performance import (
    make_bounded_failure_workload,
    make_rag_execution_metadata,
    make_timeout_workload,
    mutate_health_execution_conditions,
    mutate_saved_attempt,
    prepare_evidence_case,
    prepare_performance_comparison_case,
)
from test_support.data import common as case_data
from test_support.data.performance import (
    ALTERNATIVE_MACHINE_METADATA,
    BUDGET_REQUESTS_USERS_CASES,
    EVIDENCE_CHANGE_CASES,
    HEALTH_BATCH_CONTEXT,
    HEALTH_EXECUTION_CHANGES,
    HEALTH_EXECUTION_METADATA,
    INVALID_BATCH_FIELD_CASES,
    INVALID_SAVED_ATTEMPT_CASES,
    PERFORMANCE_COMPARISON_CASES,
    RAG_PROVENANCE_FIELDS,
)
from test_support.data.scripts.performance import (
    INCOMPATIBLE_CAPTURE_MODE_FAILS_BEFORE_EXTERNAL_SETUP_MAKEPYFILE_SOURCE,
    SELECTION_MAKEPYFILE_SOURCE,
)

pytestmark = pytest.mark.unit


@title("Concurrent batches execute the exact request budget and retain errors without retries")
def test_bounded_batch(operation_lock):
    lock = operation_lock
    calls = []

    operation = make_bounded_failure_workload(calls, lock)

    report = run_batch(operation, requests=4, users=2)
    value_checks.length(calls, 4)
    value_checks.equal(report["failed"], 1)
    value_checks.equal(
        next((r for r in report["attempts"] if r["status"] == "failed"))["error_type"],
        "TimeoutError",
    )
    validate_batch(report)
    errors.rejects(
        lambda: assertions.assert_performance_batch(report, maximum_p95=10),
        expected=AssertionError,
        match="failure rate",
    )
    assertions.assert_performance_batch(report, maximum_p95=10, maximum_failure_rate=0.25)


@pytest.mark.parametrize(("requests", "users"), BUDGET_REQUESTS_USERS_CASES)
@title("Invalid load budgets fail before making requests [{param_id}]")
def test_budget(requests, users):
    calls = []
    errors.rejects(
        lambda: run_batch(lambda: calls.append(1), requests=requests, users=users),
        expected=ValueError,
    )
    value_checks.falsy(calls)


@pytest.mark.parametrize("change", EVIDENCE_CHANGE_CASES)
@title(
    "Performance summaries reject missing attempts and inconsistent or nonfinite measurements [{param_id}]"
)
def test_evidence(change):
    report = run_batch(lambda: None, requests=2)
    altered = deepcopy(report)
    prepare_evidence_case(altered, change)
    errors.rejects(lambda: validate_batch(altered), expected=ValueError)


@title("Latency gate accepts equality and rejects a lower declared threshold")
def test_latency_gate():
    report = run_batch(lambda: None)
    value = report["latency_seconds"]["p95"]
    assertions.assert_performance_batch(report, maximum_p95=value)
    errors.rejects(
        lambda: assertions.assert_performance_batch(report, maximum_p95=value / 2),
        expected=AssertionError,
        match="p95 exceeds",
    )


@title(
    "Performance scenarios are opt-in and reject oversized selected batches before fixture setup"
)
def test_selection(framework_pytester):
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    framework_pytester.makepyfile(SELECTION_MAKEPYFILE_SOURCE)
    pytest_runs.outcomes(framework_pytester.runpytest_subprocess("-q"), skipped=2)
    result = framework_pytester.runpytest_subprocess(
        "--run-performance", "--performance-requests", "21", "-q"
    )
    value_checks.equal(result.ret, pytest.ExitCode.USAGE_ERROR)


@pytest.mark.parametrize(
    ("change", "expected"),
    PERFORMANCE_COMPARISON_CASES,
)
@title(
    "Performance baselines detect slowdown while rejecting changed or missing experiment metadata [{param_id}]"
)
def test_performance_comparison(change, expected):

    base = run_batch(lambda: None)
    base["metadata"] = case_data.fresh(HEALTH_EXECUTION_METADATA)
    current = deepcopy(base)
    prepare_performance_comparison_case(change, current)
    value_checks.equal(compare_batches(base, current)["status"], expected)


@pytest.mark.parametrize(
    "field,value",
    INVALID_BATCH_FIELD_CASES,
)
def test_saved_batch_revalidates_budget_and_numeric_types(field, value):
    report = run_batch(lambda: None)
    report[field] = value
    errors.rejects(lambda: validate_batch(report), expected=ValueError)


@pytest.mark.parametrize(
    "field",
    RAG_PROVENANCE_FIELDS,
)
def test_rag_comparison_requires_explicit_provenance(field):

    base = run_batch(lambda: None)
    base["metadata"] = make_rag_execution_metadata()
    base["metadata"].pop(field)
    value_checks.equal(compare_batches(base, deepcopy(base))["status"], "incomparable")


def test_incompatible_capture_mode_fails_before_external_setup(framework_pytester):
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    framework_pytester.makepyfile(
        INCOMPATIBLE_CAPTURE_MODE_FAILS_BEFORE_EXTERNAL_SETUP_MAKEPYFILE_SOURCE
    )
    result = framework_pytester.runpytest_subprocess(
        "--run-performance", "--performance-mode=rag", "--capture-rag", "-q"
    )
    value_checks.equal(result.ret, pytest.ExitCode.USAGE_ERROR)


@pytest.mark.parametrize("change", HEALTH_EXECUTION_CHANGES)
def test_health_comparison_requires_matching_execution_conditions(change):

    base = run_batch(lambda: None)
    base["metadata"] = case_data.fresh(ALTERNATIVE_MACHINE_METADATA)
    current = deepcopy(base)
    mutate_health_execution_conditions(base, change, current)
    value_checks.equal(compare_batches(base, current)["status"], "incomparable")


@title("Completed performance batches retain actual timeout and machine identity")
def test_completed_batches_retain_actual_timeout(tmp_path):
    report = run_batch(lambda: None)
    record_batch(report, tmp_path, **HEALTH_BATCH_CONTEXT)
    paths = list(tmp_path.glob("*.json"))
    value_checks.length(paths, 1)
    saved = json.loads(paths[0].read_text())
    value_checks.equal(saved["metadata"]["timeout"], 3)
    value_checks.truthy(saved["metadata"]["machine"])
    value_checks.excludes(report, "metadata")
    value_checks.equal(saved["failed"], 0)


@title("Failed performance batches save original attempt evidence before rejecting the gate")
def test_failed_batches_retain_attempt_evidence(tmp_path):
    report = run_batch(make_timeout_workload())
    errors.rejects(
        lambda: record_batch(report, tmp_path, **HEALTH_BATCH_CONTEXT),
        expected=AssertionError,
        match="failure rate",
    )
    paths = list(tmp_path.glob("*.json"))
    value_checks.length(paths, 1)
    saved = json.loads(paths[0].read_text())
    value_checks.equal(saved["metadata"]["timeout"], 3)
    value_checks.truthy(saved["metadata"]["machine"])
    value_checks.excludes(report, "metadata")
    value_checks.equal(saved["failed"], 1)
    value_checks.equal(saved["attempts"][0]["error"], "retained")


@pytest.mark.parametrize("change", INVALID_SAVED_ATTEMPT_CASES)
def test_malformed_saved_attempts_are_rejected_as_validation_errors(change):
    report = run_batch(lambda: None)
    mutate_saved_attempt(change, report)
    errors.rejects(lambda: validate_batch(report), expected=ValueError)
