"""Register framework plugins without eagerly loading optional browser/judge packages."""

import pytest

pytest.register_assert_rewrite("llm_testkit.assertions", "test_support.assertions")

pytest_plugins = (
        "test_support.fixtures.ci", "test_support.fixtures.conversation", "pytester",
        "llm_testkit.pytest_support.options", "llm_testkit.pytest_support.evidence", "test_support.fixtures.evidence",
        "test_support.fixtures.quality", "test_support.fixtures.performance", "test_support.fixtures.environment",
        "test_support.fixtures.installation", "test_support.fixtures.resources", "test_support.fixtures.rag")
