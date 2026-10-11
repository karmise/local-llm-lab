"""Quality gates: experimental thresholds that turn validated measurements into pass/fail checks, failing closed."""

import hashlib
import json
from copy import deepcopy

import pytest

from llm_testkit import assertions
from llm_testkit.reporting.gates import METRICS, apply_quality_gates, load_quality_gates
from llm_testkit.reporting.steps import title
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit

GATES_FILE = AUTOMATION_ROOT / "test_data/quality-gates.json"
MINIMA = {"faithfulness": 0.9, "factual_correctness": None, "context_precision": 0.8, "context_recall": 0.9}
GATED = sorted(metric for metric, minimum in MINIMA.items() if minimum is not None)


def measured_report(**values: float) -> dict:
    """A report whose fact and source checks passed and whose four metrics were measured (1.0 unless given)."""
    return {
            "schema_version":
            1,
            "status":
            "checks_passed",
            "interpretation":
            "Measured without thresholds",
            "dimensions": [{
            "name": "Facts",
            "status": "passed"}, {
            "name": "Sources",
            "status": "passed"}, *({
            "name": metric,
            "metric": metric,
            "status": "measured",
            "details": {
            "value": values.get(metric, 1.0),
            "threshold": None}} for metric in sorted(METRICS))]}


def metric_row(report: dict, metric: str) -> dict:
    return next(dimension for dimension in report["dimensions"] if dimension.get("metric") == metric)


def write_gates(tmp_path, **changes) -> object:
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(json.loads(GATES_FILE.read_text()) | changes))
    return path


@title("The reviewed gates are experimental, cover all four metrics and carry their checksum")
def test_reviewed_gates_load():
    gates = load_quality_gates(GATES_FILE)

    assert (gates["schema_version"], gates["calibration"],
            gates["version"]) == (1, "experimental", "policy-quality-gates-v2")
    assert gates["minimum_scores"] == MINIMA
    assert "Not clinically validated" in gates["rationale"]
    assert gates["sha256"] == hashlib.sha256(GATES_FILE.read_bytes()).hexdigest()


@pytest.mark.parametrize(("changes", "message"), [
        pytest.param({"schema_version": 2}, "Unsupported quality gates schema", id="schema-two"),
        pytest.param({"schema_version": True}, "Unsupported quality gates schema", id="schema-boolean"),
        pytest.param({"schema_version": None}, "Unsupported quality gates schema", id="no-schema"),
        pytest.param({"calibration": "clinically validated"}, "explicitly experimental gates only", id="clinical"),
        pytest.param({"version": ""}, "require version and rationale", id="blank-version"),
        pytest.param({"version": 1}, "require version and rationale", id="numeric-version"),
        pytest.param({"rationale": " "}, "require version and rationale", id="blank-rationale"),
        pytest.param({"minimum_scores": [0.9]}, "must define all four semantic metrics", id="minima-not-object"),
        pytest.param({"minimum_scores": MINIMA | {
        "accuracy": 0.8}}, "all four semantic metrics", id="unknown-metric"),
        pytest.param({"minimum_scores": {
        k: v
        for k, v in MINIMA.items() if k != "faithfulness"}}, "all four semantic metrics", id="missing-metric")])
@title("Invalid gate configuration is rejected rule by rule [{param_id}]")
def test_gates_reject_invalid_configuration(tmp_path, changes, message):
    with pytest.raises(ValueError, match=message):
        load_quality_gates(write_gates(tmp_path, **changes))


@pytest.mark.parametrize(
        "minimum", [
        pytest.param(float("nan"), id="nan"),
        pytest.param(True, id="boolean"),
        pytest.param(1.1, id="above-one"),
        pytest.param(-0.1, id="negative")])
@title("A threshold must be a finite score between 0 and 1 [{param_id}]")
def test_gates_reject_invalid_threshold(tmp_path, minimum):
    path = write_gates(tmp_path, minimum_scores=MINIMA | {"faithfulness": minimum})

    with pytest.raises(ValueError,
            match="Quality gate minimum for faithfulness must be a finite number between 0 and 1"):
        load_quality_gates(path)


@pytest.mark.parametrize("minimum", [pytest.param(0, id="zero"), pytest.param(1, id="one")])
@title("Thresholds at the ends of the score range are accepted [{param_id}]")
def test_gates_accept_boundary_thresholds(tmp_path, minimum):
    path = write_gates(tmp_path, minimum_scores=MINIMA | {"faithfulness": minimum})

    assert load_quality_gates(path)["minimum_scores"]["faithfulness"] == minimum


@title("Gates turn measurements into threshold checks without modifying the original report")
def test_gates_pass_measurements_at_or_above_minimum():
    report = measured_report()
    before = deepcopy(report)

    gated = apply_quality_gates(report, GATES_FILE)

    assert report == before
    assert gated["status"] == "checks_passed"
    assert gated["quality_gates"] == load_quality_gates(GATES_FILE)
    assert gated["interpretation"] == gated["quality_gates"]["rationale"]
    assert gated["dimensions"][:2] == report["dimensions"][:2]
    assert [metric_row(gated, metric) for metric in GATED] == [{
            "name": f"{metric} gate (minimum {MINIMA[metric]})",
            "metric": metric,
            "status": "passed",
            "details": {
            "value": 1.0,
            "threshold": MINIMA[metric]}} for metric in GATED]
    assert metric_row(gated, "factual_correctness") == {
            "name": "factual_correctness measurement (no threshold)",
            "metric": "factual_correctness",
            "status": "measured",
            "details": {
            "value": 1.0,
            "threshold": None}}
    assertions.assert_quality_report(gated)


@pytest.mark.parametrize("metric", GATED)
@title("A score equal to the minimum passes and an immediately lower one fails [{metric}]")
def test_gate_boundary(metric):
    at_minimum = apply_quality_gates(measured_report(**{metric: MINIMA[metric]}), GATES_FILE)
    below = apply_quality_gates(measured_report(**{metric: MINIMA[metric] - 1e-6}), GATES_FILE)

    assert (at_minimum["status"], metric_row(at_minimum, metric)["status"]) == ("checks_passed", "passed")
    assert (below["status"], metric_row(below, metric)["status"]) == ("failed", "failed")


@title("A metric without a threshold stays a measurement however low, so it cannot fail the run")
def test_ungated_metric_is_measured():
    gated = apply_quality_gates(measured_report(factual_correctness=0.0), GATES_FILE)

    assert (gated["status"], metric_row(gated, "factual_correctness")["status"]) == ("checks_passed", "measured")


@pytest.mark.parametrize(("status", "overall"), [
        pytest.param("failed", "failed", id="failed"),
        pytest.param("error", "error", id="error"),
        pytest.param("passed", "error", id="not-measured")])
@title("An ungated metric that failed, errored or was not measured is not hidden [{param_id}]")
def test_ungated_metric_keeps_failures(status, overall):
    report = measured_report()
    metric_row(report, "factual_correctness")["status"] = status

    assert apply_quality_gates(report, GATES_FILE)["status"] == overall


@title("Gates must keep at least one threshold")
def test_gates_require_a_threshold(tmp_path):
    with pytest.raises(ValueError, match="Quality gates require at least one threshold"):
        load_quality_gates(write_gates(tmp_path, minimum_scores=dict.fromkeys(MINIMA)))


@title("A low score fails with a message naming the score and the minimum")
def test_low_score_fails_with_message():
    gated = apply_quality_gates(measured_report(faithfulness=0.5), GATES_FILE)

    assert metric_row(gated, "faithfulness")["error"] == "Score 0.5 is below minimum 0.9"
    with pytest.raises(AssertionError, match="Score 0.5 is below minimum 0.9"):
        assertions.assert_quality_report(gated)


@pytest.mark.parametrize("value", [pytest.param(float("nan"), id="nan"), pytest.param(None, id="none")])
@title("A measured value that is not a quality score is rejected [{param_id}]")
def test_gate_rejects_invalid_measured_value(value):
    with pytest.raises(AssertionError, match="Expected a finite quality score"):
        apply_quality_gates(measured_report(faithfulness=value), GATES_FILE)


@title("A failed or invalid measurement stays failed or in error; a gate cannot hide it")
def test_gate_keeps_failed_and_error_dimensions():
    report = measured_report()
    metric_row(report, "faithfulness").update(status="error", error="Checksum mismatch")
    metric_row(report, "context_recall").update(status="failed", error="Low recall")

    gated = apply_quality_gates(report, GATES_FILE)

    assert gated["status"] == "error"
    assert (metric_row(gated, "faithfulness")["status"], metric_row(gated,
            "faithfulness")["error"]) == ("error", "Checksum mismatch")
    assert (metric_row(gated, "context_recall")["status"], metric_row(gated,
            "context_recall")["error"]) == ("failed", "Low recall")
    assert metric_row(gated, "faithfulness")["details"]["threshold"] is None


@title("A failure alone makes the gated report fail rather than error")
def test_failed_dimension_fails_report():
    report = measured_report()
    metric_row(report, "context_recall").update(status="failed", error="Low recall")

    assert apply_quality_gates(report, GATES_FILE)["status"] == "failed"


@title("A gated metric that was not measured is an error")
def test_gate_requires_measured_evidence():
    report = measured_report()
    metric_row(report, "faithfulness")["status"] = "passed"

    gated = apply_quality_gates(report, GATES_FILE)

    assert gated["status"] == "error"
    assert metric_row(gated, "faithfulness")["error"] == "Gate requires validated measured evidence"


@title("Every metric without evidence is reported as its own error, in name order")
def test_missing_metrics_are_errors():
    report = measured_report()
    report["dimensions"] = report["dimensions"][:2]

    gated = apply_quality_gates(report, GATES_FILE)

    assert gated["status"] == "error"
    assert gated["dimensions"][2:] == [{
            "name": f"{metric} gate",
            "metric": metric,
            "status": "error",
            "error": "Required metric evidence was not supplied"} for metric in sorted(METRICS)]


@title("Dimensions for metrics without a gate are left unchanged")
def test_ungated_metric_dimensions_are_unchanged():
    report = measured_report()
    report["dimensions"].append({"name": "answer_relevancy", "metric": "answer_relevancy", "status": "measured"})

    gated = apply_quality_gates(report, GATES_FILE)

    assert gated["dimensions"][-1] == report["dimensions"][-1]
    assert gated["status"] == "checks_passed"


@title("Duplicate metric dimensions cannot satisfy a gate")
def test_duplicate_metric_is_rejected():
    report = measured_report()
    report["dimensions"].append(deepcopy(report["dimensions"][-1]))

    with pytest.raises(ValueError, match="Duplicate quality metric dimension"):
        apply_quality_gates(report, GATES_FILE)


@title("A low measured value produces a failing pytest exit code for CI")
def test_low_score_fails_the_pytest_run(framework_pytester):
    framework_pytester.makepyfile(
            f"""
        from pathlib import Path
        from llm_testkit.assertions import assert_quality_report
        from llm_testkit.reporting.gates import apply_quality_gates

        def test_gate():
            report = {measured_report(faithfulness=0.0)!r}
            assert_quality_report(apply_quality_gates(report, Path({str(GATES_FILE)!r})))
    """)

    result = framework_pytester.runpytest_subprocess("-q")

    result.assert_outcomes(failed=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
