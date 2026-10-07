import json
from copy import deepcopy

import pytest

from llm_testkit import assertions
from llm_testkit.reporting.gates import apply_quality_gates, load_quality_gates
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import pytest_runs
from test_support.assertions import values as value_checks
from test_support.builders.quality_gates import (
    measured_report,
    prepare_configuration_case,
    prepare_fail_closed_case,
)
from test_support.data.quality_gates import (
    BOUNDARY_METRIC_CASES,
    CONFIGURATION_CHANGE_CASES,
    FAIL_CLOSED_CASES,
    GATES,
)
from test_support.data.scripts.quality_gates import render_pytest_exit_code_makepyfile_source

pytestmark = pytest.mark.unit


@title(
    "Explicit gates turn validated measurements into threshold checks without modifying the original report"
)
def test_apply_gates():
    report = measured_report()
    before = deepcopy(report)
    gated = apply_quality_gates(report, GATES)
    assertions.assert_quality_report(gated)
    value_checks.equal(gated["status"], "checks_passed")
    value_checks.equal(report, before)
    value_checks.equal(gated["quality_gates"]["calibration"], "experimental")
    value_checks.all_true((d["status"] == "passed" for d in gated["dimensions"]))


@pytest.mark.parametrize("metric,change,expected_status", FAIL_CLOSED_CASES)
@title("Quality gates fail closed for missing, invalid or low-scoring metrics [{param_id}]")
def test_fail_closed(metric, change, expected_status):
    report = measured_report()
    row = next(d for d in report["dimensions"] if d.get("metric") == metric)
    prepare_fail_closed_case(change, report, row)
    gated = apply_quality_gates(report, GATES)
    value_checks.equal(gated["status"], expected_status)
    errors.rejects(lambda: assertions.assert_quality_report(gated), expected=AssertionError)


@pytest.mark.parametrize("metric", BOUNDARY_METRIC_CASES)
@title("Quality threshold equality passes and an immediately lower measurement fails [{param_id}]")
def test_boundary(metric):
    report = measured_report()
    row = next(d for d in report["dimensions"] if d.get("metric") == metric)
    row["details"]["value"] = load_quality_gates(GATES)["minimum_scores"][metric]
    value_checks.equal(apply_quality_gates(report, GATES)["status"], "checks_passed")
    row["details"]["value"] -= 1e-6
    value_checks.equal(apply_quality_gates(report, GATES)["status"], "failed")


@pytest.mark.parametrize("change", CONFIGURATION_CHANGE_CASES)
@title(
    "Gate configuration requires all metrics, finite thresholds and an explicit calibration boundary [{param_id}]"
)
def test_configuration(tmp_path, change):
    config = json.loads(GATES.read_text())
    prepare_configuration_case(change, config)
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(config))
    errors.rejects(lambda: load_quality_gates(path), expected=(ValueError, AssertionError))


@title("Duplicate metric dimensions cannot satisfy quality gates")
def test_duplicate_metric():
    report = measured_report()
    report["dimensions"].append(deepcopy(report["dimensions"][-1]))
    errors.rejects(
        lambda: apply_quality_gates(report, GATES), expected=ValueError, match="Duplicate"
    )


@title("A low measured value produces a failing pytest exit code for CI")
def test_pytest_exit_code(framework_pytester):
    framework_pytester.makepyfile(render_pytest_exit_code_makepyfile_source())
    result = framework_pytester.runpytest_subprocess("-q")
    pytest_runs.outcomes(result, failed=1)
    value_checks.equal(result.ret, pytest.ExitCode.TESTS_FAILED)
