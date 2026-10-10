"""Commit-bound saved CI inputs; this layer never calls the application or judge."""

import pytest

from llm_testkit.ci.benchmark import validate_ci_benchmark


@pytest.fixture
def saved_ci_benchmark_report(pytestconfig, automation_root):
    options = {
            name: pytestconfig.getoption(name)
            for name in ("benchmark_report", "ci_profile", "ci_models", "ci_revision")}
    if all(value is None for value in options.values()):
        pytest.skip("Supply the saved benchmark and explicit CI identity")
    if any(value is None for value in options.values()):
        pytest.fail(
                "CI reporting requires --benchmark-report, --ci-profile, --ci-models and --ci-revision", pytrace=False)
    return validate_ci_benchmark(
            automation_root, options["benchmark_report"], revision=options["ci_revision"],
            profile=options["ci_profile"], models=options["ci_models"])
