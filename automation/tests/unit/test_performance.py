import json
from copy import deepcopy
from threading import Lock

import pytest

from llm_testkit import assertions
from llm_testkit.performance.comparison import compare_batches
from llm_testkit.performance.reporting import record_batch
from llm_testkit.performance.runner import run_batch, validate_batch
from llm_testkit.reporting.steps import title
from test_support.builders.performance import (
    make_bounded_failure_workload,
    make_timeout_workload,
    mutate_health_execution_conditions,
    mutate_saved_attempt,
    prepare_evidence_case,
    prepare_performance_comparison_case,
)
from test_support.data.performance import (
    BUDGET_REQUESTS_USERS_CASES,
    EVIDENCE_CHANGE_CASES,
    HEALTH_BATCH_CONTEXT,
    HEALTH_EXECUTION_CHANGES,
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
def test_bounded_batch():
    lock = Lock()
    calls = []

    operation = make_bounded_failure_workload(calls, lock)

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
    PERFORMANCE_COMPARISON_CASES,
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
    INVALID_BATCH_FIELD_CASES,
)
def test_saved_batch_revalidates_budget_and_numeric_types(field, value):
    report = run_batch(lambda: None)
    report[field] = value
    with pytest.raises(ValueError):
        validate_batch(report)


@pytest.mark.parametrize(
    "field",
    RAG_PROVENANCE_FIELDS,
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


@pytest.mark.parametrize("change", HEALTH_EXECUTION_CHANGES)
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
    mutate_health_execution_conditions(base, change, current)
    assert compare_batches(base, current)["status"] == "incomparable"


@title("Completed performance batches retain actual timeout and machine identity")
def test_completed_batches_retain_actual_timeout(tmp_path):
    report = run_batch(lambda: None)
    record_batch(report, tmp_path, **HEALTH_BATCH_CONTEXT)
    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1
    saved = json.loads(paths[0].read_text())
    assert saved["metadata"]["timeout"] == 3
    assert saved["metadata"]["machine"]
    assert "metadata" not in report
    assert saved["failed"] == 0


@title("Failed performance batches save original attempt evidence before rejecting the gate")
def test_failed_batches_retain_attempt_evidence(tmp_path):
    report = run_batch(make_timeout_workload())
    with pytest.raises(AssertionError, match="failure rate"):
        record_batch(report, tmp_path, **HEALTH_BATCH_CONTEXT)
    paths = list(tmp_path.glob("*.json"))
    assert len(paths) == 1
    saved = json.loads(paths[0].read_text())
    assert saved["metadata"]["timeout"] == 3
    assert saved["metadata"]["machine"]
    assert "metadata" not in report
    assert saved["failed"] == 1
    assert saved["attempts"][0]["error"] == "retained"


@pytest.mark.parametrize("change", INVALID_SAVED_ATTEMPT_CASES)
def test_malformed_saved_attempts_are_rejected_as_validation_errors(change):
    report = run_batch(lambda: None)
    mutate_saved_attempt(change, report)
    with pytest.raises(ValueError):
        validate_batch(report)
