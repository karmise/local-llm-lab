import json
from copy import deepcopy

import pytest

from llm_testkit import assertions
from llm_testkit.reporting.gates import apply_quality_gates, load_quality_gates
from llm_testkit.reporting.steps import title
from test_support.builders.quality_gates import (
    check_fail_closed_step_5,
    measured_report,
    prepare_configuration_case,
    prepare_fail_closed_case,
)
from test_support.data.quality_gates import (
    BOUNDARY_METRIC_CASES,
    CONFIGURATION_CHANGE_CASES,
    FAIL_CLOSED_CHANGE_CASES,
    FAIL_CLOSED_METRIC_CASES,
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
    assert gated["status"] == "checks_passed"
    assert report == before
    assert gated["quality_gates"]["calibration"] == "experimental"
    assert all(d["status"] == "passed" for d in gated["dimensions"])


@pytest.mark.parametrize("metric", FAIL_CLOSED_METRIC_CASES)
@pytest.mark.parametrize("change", FAIL_CLOSED_CHANGE_CASES)
@title("Quality gates fail closed for missing, invalid or low-scoring metrics [{param_id}]")
def test_fail_closed(metric, change):
    report = measured_report()
    row = next(d for d in report["dimensions"] if d.get("metric") == metric)
    prepare_fail_closed_case(change, report, row)
    gated = apply_quality_gates(report, GATES)
    check_fail_closed_step_5(change, gated)
    with pytest.raises(AssertionError):
        assertions.assert_quality_report(gated)


@pytest.mark.parametrize("metric", BOUNDARY_METRIC_CASES)
@title("Quality threshold equality passes and an immediately lower measurement fails [{param_id}]")
def test_boundary(metric):
    report = measured_report()
    row = next(d for d in report["dimensions"] if d.get("metric") == metric)
    row["details"]["value"] = load_quality_gates(GATES)["minimum_scores"][metric]
    assert apply_quality_gates(report, GATES)["status"] == "checks_passed"
    row["details"]["value"] -= 1e-6
    assert apply_quality_gates(report, GATES)["status"] == "failed"


@pytest.mark.parametrize("change", CONFIGURATION_CHANGE_CASES)
@title(
    "Gate configuration requires all metrics, finite thresholds and an explicit calibration boundary [{param_id}]"
)
def test_configuration(tmp_path, change):
    config = json.loads(GATES.read_text())
    prepare_configuration_case(change, config)
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(config))
    with pytest.raises((ValueError, AssertionError)):
        load_quality_gates(path)


@title("Duplicate metric dimensions cannot satisfy quality gates")
def test_duplicate_metric():
    report = measured_report()
    report["dimensions"].append(deepcopy(report["dimensions"][-1]))
    with pytest.raises(ValueError, match="Duplicate"):
        apply_quality_gates(report, GATES)


@title("A low measured value produces a failing pytest exit code for CI")
def test_pytest_exit_code(framework_pytester):
    framework_pytester.makepyfile(render_pytest_exit_code_makepyfile_source())
    result = framework_pytester.runpytest_subprocess("-q")
    result.assert_outcomes(failed=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
