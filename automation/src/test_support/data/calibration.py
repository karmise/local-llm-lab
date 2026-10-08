"""Named parameter cases for calibration scenarios."""

RUNNER_REJECTS_EMPTY_OR_EXCESSIVE_CONTROL_RUNS_COUNT_CASES = [0, 4]

CONTROL_LOADER_REJECTS_WRONG_CONTEXT_OR_BAD_LABELS_CHANGE_CASES = ["context", "score", "duplicate", "empty"]

# Input for test_control_runner_preserves_mismatch_and_error_then_continues
ORIGINAL_INPUT = {"response": "Original application answer", "retrieved_contexts": ["Policy"]}

# Input for test_faithful_incomplete_answer_still_fails_required_fact_check
FACTS_23_WORKING_DAYS_12_CALENDAR_DAYS_INPUT = {
        "23 working days": "\\b23\\s+working\\s+days\\b",
        "12 calendar days": "\\b12\\s+calendar\\s+days\\b"}
