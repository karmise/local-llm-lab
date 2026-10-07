"""Named parameter cases for performance scenarios."""

BUDGET_REQUESTS_USERS_CASES = [(0, 1), (21, 1), (2, 3), (5, 5), (True, 1)]

EVIDENCE_CHANGE_CASES = ["attempts", "latency", "rps", "failed", "nan"]

PERFORMANCE_COMPARISON_CHANGE_EXPECTED_CASES = [
    ("none", "passed"),
    ("slower", "regression"),
    ("machine", "incomparable"),
    ("missing", "incomparable"),
]

SAVED_BATCH_REVALIDATES_BUDGET_AND_NUMERIC_TYPES_FIELD_VALUE_CASES = [
    ("users", True),
    ("users", 5),
    ("requests", True),
    ("schema_version", True),
    ("wall_seconds", True),
]

RAG_COMPARISON_REQUIRES_EXPLICIT_PROVENANCE_FIELD_CASES = [
    "timeout",
    "policy_sha256",
    "golden_dataset_sha256",
    "case_id",
    "model_digest",
    "configuration",
]

HEALTH_COMPARISON_REQUIRES_MATCHING_EXECUTION_CONDITIONS_CHANGE_CASES = [
    "timeout",
    "machine",
    "empty-machine",
    "boolean-warmup",
]

RECORDED_BATCHES_RETAIN_ACTUAL_TIMEOUT_AND_FAILURES_FAILURE_CASES = [False, True]

MALFORMED_SAVED_ATTEMPTS_ARE_REJECTED_AS_VALIDATION_ERRORS_CHANGE_CASES = [
    "row",
    "index",
    "status",
    "elapsed",
    "wall",
    "failed",
    "rps",
]
