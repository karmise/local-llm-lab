"""Named scenario preparation and test doubles."""

import pytest

from llm_testkit.performance.reporting import record_batch


def make_operation_stub(calls, lock):
    def operation():
        with lock:
            calls.append(1)
            index = len(calls)
        if index == 2:
            raise TimeoutError("Timeout")

    return operation


def prepare_evidence_case(altered, change):
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


def prepare_performance_comparison_case(change, current):
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


def prepare_health_comparison_requires_matching_execution_conditions_case(base, change, current):
    if change == "empty-machine":
        base["metadata"]["machine"] = current["metadata"]["machine"] = ""
    elif change == "boolean-warmup":
        current["metadata"]["warmup_requests"] = False
    else:
        current["metadata"][change] = 10 if change == "timeout" else "second"


def make_operation_stub_2(failure):
    def operation():
        if failure:
            raise TimeoutError("retained")

    return operation


def check_recorded_batches_retain_actual_timeout_and_failures_outcome(
    failure, kwargs, report, tmp_path
):
    if failure:
        with pytest.raises(AssertionError, match="failure rate"):
            record_batch(report, tmp_path, **kwargs)
    else:
        record_batch(report, tmp_path, **kwargs)


def check_recorded_batches_retain_actual_timeout_and_failures_outcome_2(failure, saved):
    if failure:
        assert saved["attempts"][0]["error"] == "retained"


def prepare_malformed_saved_attempts_are_rejected_as_validation_errors_case(change, report):
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
