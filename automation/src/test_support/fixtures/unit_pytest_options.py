"""Fixtures scoped to the associated unit-test module."""

import pytest


@pytest.fixture
def runner(framework_pytester: pytest.Pytester) -> pytest.Pytester:
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    return framework_pytester
