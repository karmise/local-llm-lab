"""Named parameter cases for performance scenarios."""

BUDGET_REQUESTS_USERS_CASES = [(0, 1), (21, 1), (2, 3), (5, 5), (True, 1)]

EVIDENCE_CHANGE_CASES = ["attempts", "latency", "rps", "failed", "nan"]

PERFORMANCE_COMPARISON_CASES = [
    ("none", "passed"),
    ("slower", "regression"),
    ("machine", "incomparable"),
    ("missing", "incomparable"),
]

INVALID_BATCH_FIELD_CASES = [
    ("users", True),
    ("users", 5),
    ("requests", True),
    ("schema_version", True),
    ("wall_seconds", True),
]

RAG_PROVENANCE_FIELDS = [
    "timeout",
    "policy_sha256",
    "golden_dataset_sha256",
    "case_id",
    "model_digest",
    "configuration",
]

HEALTH_EXECUTION_CHANGES = [
    "timeout",
    "machine",
    "empty-machine",
    "boolean-warmup",
]

HEALTH_BATCH_CONTEXT = {
    "base_url": "http://localhost",
    "timeout": 3,
    "maximum_p95": 10,
    "metadata": {"workload": "health"},
}

INVALID_SAVED_ATTEMPT_CASES = [
    "row",
    "index",
    "status",
    "elapsed",
    "wall",
    "failed",
    "rps",
]


# Input for test_performance_comparison
HEALTH_EXECUTION_METADATA = {
    "workload": "health",
    "system": "test",
    "machine": "test-machine",
    "python": "3.12",
    "base_url": "http://localhost",
    "warmup_requests": 0,
    "timeout": 5,
}


# Input for test_health_comparison_requires_matching_execution_conditions
ALTERNATIVE_MACHINE_METADATA = {
    "workload": "health",
    "system": "test",
    "machine": "first",
    "python": "3.12",
    "base_url": "http://localhost",
    "warmup_requests": 0,
    "timeout": 5,
}
