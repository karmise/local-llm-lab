"""Named scenario preparation and test doubles."""


def make_bounded_failure_workload(calls, lock):
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


def mutate_health_execution_conditions(base, change, current):
    if change == "empty-machine":
        base["metadata"]["machine"] = current["metadata"]["machine"] = ""
    elif change == "boolean-warmup":
        current["metadata"]["warmup_requests"] = False
    else:
        current["metadata"][change] = 10 if change == "timeout" else "second"


def make_timeout_workload():
    def operation():
        raise TimeoutError("retained")

    return operation


def mutate_saved_attempt(change, report):
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


def make_rag_execution_metadata():
    """Build input for test_rag_comparison_requires_explicit_provenance."""
    return {
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
