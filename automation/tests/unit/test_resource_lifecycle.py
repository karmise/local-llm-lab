"""Exercise resource lifecycles through pytest, including setup and teardown failures."""

import json

import pytest

from test_support.assertions import pytest_runs
from test_support.assertions import values as value_checks
from test_support.builders.resource_lifecycle import (
    prepare_resources_are_cleaned_up_after_each_failure_case,
)
from test_support.data.resource_lifecycle import (
    RESOURCES_ARE_CLEANED_UP_AFTER_EACH_FAILURE_FAILURE_FAILED_ERRORS_CLEANUP_CASES,
)
from test_support.data.scripts.resource_lifecycle import (
    render_resource_conftest,
    render_resource_test_source,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("failure", "failed", "errors", "cleanup"),
    RESOURCES_ARE_CLEANED_UP_AFTER_EACH_FAILURE_FAILURE_FAILED_ERRORS_CLEANUP_CASES,
)
def test_resources_are_cleaned_up_after_each_failure(
    framework_pytester: pytest.Pytester,
    failure: str,
    failed: int,
    errors: int,
    cleanup: list[str],
) -> None:
    runner = framework_pytester
    runner.makeconftest(render_resource_conftest(failure))
    runner.makepyfile(render_resource_test_source(failure))

    result = runner.runpytest_subprocess("-q", "--tb=short")

    pytest_runs.outcomes(result, passed=int(failure == "none"), failed=failed, errors=errors)
    value_checks.equal(json.loads((runner.path / "cleanup.json").read_text()), cleanup)
    prepare_resources_are_cleaned_up_after_each_failure_case(failure, result)
