"""Scenario data builders and deterministic test doubles."""

from unittest.mock import Mock

import pytest

from llm_testkit.evaluation.calibration import load_controls


def _case() -> dict:
    return {
        "id": "mixed",
        "response": "Leave is 23 days. Gym reimbursement is 5000.",
        "expected_score": 0.5,
        "claims": [{"pattern": "23", "verdict": 1}, {"pattern": "5000", "verdict": 0}],
    }


def _result() -> dict:
    return {
        "value": 0.5,
        "verdicts": [
            {"statement": "Leave is 23 days.", "verdict": 1},
            {"statement": "Gym reimbursement is 5000.", "verdict": 0},
        ],
    }


def prepare_control_loader_rejects_wrong_context_or_bad_labels_case(change, controls):
    if change == "score":
        controls["cases"][0]["expected_score"] = 1.0
    elif change == "duplicate":
        controls["cases"].append(_case())
    elif change == "empty":
        controls["cases"] = []


def make_fake_score_stub(observed_responses):
    async def fake_score(sample: dict, judge: object) -> dict:
        observed_responses.append(sample["response"])
        if len(observed_responses) == 3:
            raise ValueError("Truncated judge response")
        return _result() if len(observed_responses) == 1 else {**_result(), "value": 1.0}

    return fake_score


def make_factory_stub(judges):
    def factory() -> Mock:
        judge = Mock(calls=[])
        judges.append(judge)
        return judge

    return factory


def prepare_control_loader_rejects_wrong_context_or_bad_labels_step_5(change, path):
    with pytest.raises(ValueError):
        load_controls(path, ["Other document" if change == "context" else "Policy"])
