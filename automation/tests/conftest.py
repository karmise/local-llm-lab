"""Register framework plugins without eagerly loading optional browser/judge packages."""

import pytest

pytest.register_assert_rewrite("llm_testkit.assertions")

pytest_plugins = (
    "llm_testkit.pytest_support.conversation",
    "pytester",
    "llm_testkit.pytest_support.options",
    "llm_testkit.pytest_support.evidence",
    "llm_testkit.pytest_support.environment",
    "llm_testkit.pytest_support.resources",
    "llm_testkit.pytest_support.rag",
)
