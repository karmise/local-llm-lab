"""Shared-check failures and diagnostics remain visible to pytest and CI."""

import pytest

from llm_testkit.reporting.steps import title
from test_support.assertions import pytest_runs
from test_support.data.scripts.unit_expectations import EXPECTATION_FAILURE_SOURCE, EXPECTED_FAILURE_MESSAGES

pytestmark = pytest.mark.unit


@title("Shared unit expectations reject wrong values, error types and messages while retaining exceptions")
def test_expectation_failures_remain_visible_to_pytest(framework_pytester):
    framework_pytester.makepyfile(EXPECTATION_FAILURE_SOURCE)
    result = framework_pytester.runpytest_subprocess("-q", "--tb=short")
    pytest_runs.outcomes(result, passed=1, failed=3)
    result.stdout.fnmatch_lines(EXPECTED_FAILURE_MESSAGES)
