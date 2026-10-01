import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from llm_testkit import assertions
from llm_testkit.evaluation.calibration import evaluate_controls, load_controls, select_controls

pytestmark = pytest.mark.unit


def test_unknown_control_selection_fails_before_model_calls() -> None:
    with pytest.raises(ValueError, match="Unknown controls"):
        select_controls([_case()], ["typo"])


def test_targeted_selection_deduplicates_controls() -> None:
    selected = select_controls([_case(), {**_case(), "id": "other"}], ["mixed", "mixed"])
    assertions.assert_field_equals({"ids": [case["id"] for case in selected]}, "ids", ["mixed"])


@pytest.mark.parametrize("count", [0, 4])
def test_runner_rejects_empty_or_excessive_control_runs(count: int) -> None:
    factory = Mock()
    with pytest.raises(ValueError, match="between one and three"):
        asyncio.run(evaluate_controls({}, [_case()] * count, factory))
    assertions.assert_field_equals({"calls": factory.call_count}, "calls", 0)


def _case() -> dict:
    return {
        "id": "mixed", "response": "Leave is 23 days. Gym reimbursement is 5000.",
        "expected_score": 0.5,
        "claims": [{"pattern": "23", "verdict": 1}, {"pattern": "5000", "verdict": 0}],
    }


def _result() -> dict:
    return {"value": 0.5, "verdicts": [
        {"statement": "Leave is 23 days.", "verdict": 1},
        {"statement": "Gym reimbursement is 5000.", "verdict": 0},
    ]}


def test_same_score_with_reversed_verdicts_fails_control_check() -> None:
    result = _result()
    result["verdicts"][0]["verdict"] = 0
    result["verdicts"][1]["verdict"] = 1
    with pytest.raises(AssertionError):
        assertions.assert_calibration_result(result, expected_score=0.5, claims=_case()["claims"])


def test_missing_claim_fails_even_when_score_matches() -> None:
    result = _result()
    result["verdicts"].pop()
    with pytest.raises(AssertionError, match="number of claims"):
        assertions.assert_calibration_result(result, expected_score=0.5, claims=_case()["claims"])


def test_two_expected_claims_cannot_match_one_combined_statement() -> None:
    result = _result()
    result["verdicts"][0]["statement"] = "Leave 23; gym 5000."
    result["verdicts"][1]["statement"] = "Other claim."
    with pytest.raises(AssertionError, match="distinct"):
        assertions.assert_calibration_result(result, expected_score=0.5, claims=_case()["claims"])


@pytest.mark.parametrize("change", ["context", "score", "duplicate", "budget"])
def test_control_loader_rejects_wrong_context_or_bad_labels(tmp_path: Path, change: str) -> None:
    controls = {"schema_version": 1, "required_context_fragments": ["Policy"], "cases": [_case()]}
    if change == "score":
        controls["cases"][0]["expected_score"] = 1.0
    elif change == "duplicate":
        controls["cases"].append(_case())
    elif change == "budget":
        controls["cases"] = [_case()] * 4
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(controls))
    with pytest.raises(ValueError):
        load_controls(path, ["Other document" if change == "context" else "Policy"])


def test_control_runner_preserves_mismatch_and_error_then_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = [_case(), {**_case(), "id": "mismatch"}, {**_case(), "id": "error"}]
    original = {"response": "Original application answer", "retrieved_contexts": ["Policy"]}
    observed_responses = []

    async def fake_score(sample: dict, judge: object) -> dict:
        observed_responses.append(sample["response"])
        if len(observed_responses) == 3:
            raise ValueError("Truncated judge response")
        return _result() if len(observed_responses) == 1 else {**_result(), "value": 1.0}

    monkeypatch.setattr("llm_testkit.evaluation.calibration.score_sample", fake_score)
    judges = []

    def factory() -> Mock:
        judge = Mock(calls=[])
        judges.append(judge)
        return judge

    results = asyncio.run(evaluate_controls(original, cases, factory))
    assertions.assert_field_equals({"statuses": [r["status"] for r in results]}, "statuses", ["matched", "mismatch", "error"])
    assertions.assert_field_equals(original, "response", "Original application answer")
    assertions.assert_field_equals({"judges": len(judges)}, "judges", 3)
    assertions.assert_field_equals(results[2]["error"], "type", "ValueError")
