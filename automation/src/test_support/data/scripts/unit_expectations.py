"""Deliberate failures verify shared checks cannot silently accept incorrect evidence."""

EXPECTATION_FAILURE_SOURCE = """
    from test_support.assertions import values, errors

    def test_wrong_score():
        values.equal(0.2, 0.9)

    def test_wrong_exception_type():
        def operation():
            raise RuntimeError("budget")
        errors.rejects(operation, expected=ValueError, match="budget")

    def test_wrong_exception_message():
        def operation():
            raise ValueError("truncated")
        errors.rejects(operation, expected=ValueError, match="budget")

    def test_original_exception_is_preserved():
        original = SystemExit(2)
        def operation():
            raise original
        caught = errors.rejects(operation, expected=SystemExit)
        assert caught.value is original
        assert caught.value.code == 2
"""

EXPECTED_FAILURE_MESSAGES = [
    "*Expected 0.9, got 0.2*",
    "*RuntimeError: budget*",
    "*Regex pattern did not match*",
]
