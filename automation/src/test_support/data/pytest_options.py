"""Named parameter cases for pytest_options scenarios."""

DESELECTED_LIVE_TESTS_DO_NOT_VALIDATE_LIVE_OPTIONS_SELECTOR_CASES = [("-m", "unit"), ("-k", "offline")]

INVALID_MATRIX_OPTIONS_ARE_USAGE_ERRORS_ARGUMENTS_CASES = [("--rag-repeat", "0"), ("--rag-repeat", "oops"),
        ("--rag-model", " ")]
