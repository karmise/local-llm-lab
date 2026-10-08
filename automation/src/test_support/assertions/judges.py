"""Independent checks for score, judge budgets and retained service failures."""

from collections.abc import Mapping
from typing import Any

import pytest

from test_support.assertions import errors, values
from test_support.builders.judge_scenarios import FailedServiceScenario, JudgeScenario, RelevanceScenario


def score_matches(scenario: JudgeScenario, result: Mapping[str, Any]) -> None:
    values.equal(result["value"], scenario.expected_score)
    values.equal(scenario.client.structured_chat.call_count, scenario.maximum_calls)
    if scenario.check_options:
        values.equal(scenario.client.structured_chat.call_args.kwargs["options"], scenario.judge.options)


def budget_is_exhausted(scenario: JudgeScenario) -> None:
    errors.rejects(
            lambda: scenario.judge.generate("An additional request must be rejected", object), expected=ValueError,
            match="budget")
    # Budget rejection must happen before another transport call.
    values.equal(scenario.client.structured_chat.call_count, scenario.maximum_calls)


def relevance_matches(scenario: RelevanceScenario, result: Mapping[str, Any]) -> None:
    values.equal(result, scenario.expected)
    values.length(scenario.precision.calls, 3)
    values.length(scenario.recall.calls, 1)
    values.equal(result["context_precision"], pytest.approx(5 / 6))


def failure_evidence_is_retained(scenario: FailedServiceScenario, report: Mapping[str, Any]) -> None:
    values.equal(report["status"], "error")
    values.equal(scenario.transport.close.call_count, 1)
    if scenario.judge is not None:
        values.equal(report["judge_calls"], scenario.judge.calls)
    else:
        values.equal(report["precision_calls"], scenario.precision.calls)
        values.equal(report["recall_calls"], scenario.recall.calls)
        values.equal(report["judge_configuration"]["maximum_calls"], 2)
