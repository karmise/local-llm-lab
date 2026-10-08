"""Domain expectations for correctness unit scenarios."""

from collections.abc import Mapping
from typing import Any

from llm_testkit.evaluation.correctness import check_control, check_correctness_evidence, validate_control
from test_support.assertions import errors, values
from test_support.builders.correctness import CorrectnessEvidenceScenario, IncompleteControlScenario
from test_support.data.correctness import CASE as CASE
from test_support.data.correctness import DATASET


def check_control_expectations(cases):
    values.length(cases, 4)
    values.length({c["id"] for c in cases}, 4)
    for control in cases:
        validate_control(control)
        assert control["case_id"] == CASE.id


def rejects_unbound_evidence(scenario: CorrectnessEvidenceScenario) -> None:
    errors.rejects(
            lambda: check_correctness_evidence(scenario.evidence, "sample", scenario.sample, DATASET),
            expected=ValueError)


def low_score_remains_a_measurement(report: Mapping[str, Any]) -> None:
    values.length(report["dimensions"], 4)
    values.equal(report["status"], "checks_passed")
    values.equal(report["dimensions"][-1]["details"]["value"], 0.0)


def invalid_correctness_does_not_invalidate_faithfulness(report: Mapping[str, Any]) -> None:
    values.equal(report["dimensions"][-1]["status"], "error")
    values.equal(report["dimensions"][2]["status"], "measured")


def rejects_incorrect_control_labels(scenario: IncompleteControlScenario) -> None:
    errors.rejects(lambda: check_control(scenario.result, scenario.control), expected=ValueError)
