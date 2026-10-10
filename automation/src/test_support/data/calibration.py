"""Named parameter cases for calibration scenarios."""

RUNNER_REJECTS_EMPTY_OR_EXCESSIVE_CONTROL_RUNS_COUNT_CASES = [0, 4]

CONTROL_LOADER_REJECTS_WRONG_CONTEXT_OR_BAD_LABELS_CHANGE_CASES = ["context", "score", "duplicate", "empty"]

# Input for test_control_runner_preserves_mismatch_and_error_then_continues
ORIGINAL_INPUT = {"response": "Original application answer", "retrieved_contexts": ["Policy"]}

# Input for test_faithful_incomplete_answer_still_fails_required_fact_check
FACTS_23_WORKING_DAYS_12_CALENDAR_DAYS_INPUT = {
        "23 working days": "\\b23\\s+working\\s+days\\b",
        "12 calendar days": "\\b12\\s+calendar\\s+days\\b"}

NUMBER_SPELLINGS = (("THIRTY working days; two calendar days", "30 working days; 2 calendar days"),
        ("twenty-three working days", "23 working days"), ("twenty three", "23"), ("thirty one",
        "31"), ("thirty thousand", "thirty thousand"), ("one hundred and thirty",
        "one hundred and thirty"), ("thirty point two", "thirty point two"))
LEAVE_CONTROL_STATEMENT = "Each employee receives thirty working days of paid leave per year."
NOTICE_CONTROL_STATEMENT = "A leave request must be submitted at least two calendar days before the leave starts."
CHANGED_NUMBER_CLAIMS = (
        "Each employee receives twenty working days of paid leave per year.",
        "Each employee receives thirty calendar days of paid leave per year.",
        "Each employee receives thirty thousand working days of paid leave per year.")
