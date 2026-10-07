import json
from copy import deepcopy

import pytest

from llm_testkit import assertions
from llm_testkit.reporting.gates import METRICS, apply_quality_gates, load_quality_gates
from llm_testkit.reporting.steps import title
from test_support.builders.quality_gates import GATES, measured_report

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


@pytest.mark.parametrize("metric", sorted(METRICS))
@pytest.mark.parametrize("change", ["low", "missing", "invalid", "not-measured"])
@title("Quality gates fail closed for missing, invalid or low-scoring metrics [{param_id}]")
def test_fail_closed(metric, change):
    report = measured_report()
    row = next(d for d in report["dimensions"] if d.get("metric") == metric)
    if change == "low":
        row["details"]["value"] = 0.0
    elif change == "missing":
        report["dimensions"].remove(row)
    elif change == "invalid":
        row.update(status="error", error="Checksum mismatch")
    else:
        row["status"] = "passed"
    gated = apply_quality_gates(report, GATES)
    assert gated["status"] == ("failed" if change == "low" else "error")
    with pytest.raises(AssertionError):
        assertions.assert_quality_report(gated)


@pytest.mark.parametrize("metric", sorted(METRICS))
@title("Quality threshold equality passes and an immediately lower measurement fails [{param_id}]")
def test_boundary(metric):
    report = measured_report()
    row = next(d for d in report["dimensions"] if d.get("metric") == metric)
    row["details"]["value"] = load_quality_gates(GATES)["minimum_scores"][metric]
    assert apply_quality_gates(report, GATES)["status"] == "checks_passed"
    row["details"]["value"] -= 1e-6
    assert apply_quality_gates(report, GATES)["status"] == "failed"


@pytest.mark.parametrize(
    "change", ["missing", "unknown", "nan", "boolean", "range", "clinical", "version"]
)
@title(
    "Gate configuration requires all metrics, finite thresholds and an explicit calibration boundary [{param_id}]"
)
def test_configuration(tmp_path, change):
    config = json.loads(GATES.read_text())
    if change == "missing":
        config["minimum_scores"].pop("faithfulness")
    elif change == "unknown":
        config["minimum_scores"]["accuracy"] = 0.8
    elif change in ("nan", "boolean", "range"):
        config["minimum_scores"]["faithfulness"] = {
            "nan": float("nan"),
            "boolean": True,
            "range": 1.1,
        }[change]
    elif change == "clinical":
        config["calibration"] = "clinically validated"
    else:
        config["version"] = ""
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
    framework_pytester.makepyfile(f"""
        from pathlib import Path
        from llm_testkit.reporting.gates import apply_quality_gates
        from llm_testkit.assertions import assert_quality_report
        def test_gate():
            report = {measured_report()!r}
            report['dimensions'][2]['details']['value'] = 0.0
            assert_quality_report(apply_quality_gates(report, Path({str(GATES)!r})))
    """)
    result = framework_pytester.runpytest_subprocess("-q")
    result.assert_outcomes(failed=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
