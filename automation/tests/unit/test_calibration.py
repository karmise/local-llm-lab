import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.evaluation.calibration import evaluate_controls, load_controls, select_controls
from llm_testkit.reporting.steps import title
from test_support.builders.calibration import _case, _result

pytestmark = pytest.mark.unit


@title("Unknown calibration control is rejected before model calls")
def test_unknown_control_selection_fails_before_model_calls() -> None:
    with pytest.raises(ValueError, match="Unknown controls"):
        select_controls([_case()], ["typo"])


@title("Targeted calibration selection removes duplicate controls")
def test_targeted_selection_deduplicates_controls() -> None:
    selected = select_controls([_case(), {**_case(), "id": "other"}], ["mixed", "mixed"])
    assert [case["id"] for case in selected] == ["mixed"]


@pytest.mark.parametrize("count", [0, 4])
@title("Calibration runner rejects empty or excessive control batches [{param_id}]")
def test_runner_rejects_empty_or_excessive_control_runs(count: int) -> None:
    factory = Mock()
    with pytest.raises(ValueError, match="between one and three"):
        asyncio.run(evaluate_controls({}, [_case()] * count, factory))
    assert factory.call_count == 0


@title("Calibration rejects reversed claim verdicts even when the score matches")
def test_same_score_with_reversed_verdicts_fails_control_check() -> None:
    result = _result()
    result["verdicts"][0]["verdict"] = 0
    result["verdicts"][1]["verdict"] = 1
    with pytest.raises(AssertionError):
        assertions.assert_calibration_result(result, expected_score=0.5, claims=_case()["claims"])


@title("Calibration rejects a missing claim even when the score matches")
def test_missing_claim_fails_even_when_score_matches() -> None:
    result = _result()
    result["verdicts"].pop()
    with pytest.raises(AssertionError, match="number of claims"):
        assertions.assert_calibration_result(result, expected_score=0.5, claims=_case()["claims"])


@title("Two expected calibration claims cannot share one combined judge statement")
def test_two_expected_claims_cannot_match_one_combined_statement() -> None:
    result = _result()
    result["verdicts"][0]["statement"] = "Leave 23; gym 5000."
    result["verdicts"][1]["statement"] = "Other claim."
    with pytest.raises(AssertionError, match="distinct"):
        assertions.assert_calibration_result(result, expected_score=0.5, claims=_case()["claims"])


@pytest.mark.parametrize("change", ["context", "score", "duplicate", "empty"])
@title("Calibration loader rejects unrelated context and invalid control labels [{param_id}]")
def test_control_loader_rejects_wrong_context_or_bad_labels(tmp_path: Path, change: str) -> None:
    controls = {"schema_version": 1, "required_context_fragments": ["Policy"], "cases": [_case()]}
    if change == "score":
        controls["cases"][0]["expected_score"] = 1.0
    elif change == "duplicate":
        controls["cases"].append(_case())
    elif change == "empty":
        controls["cases"] = []
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(controls))
    with pytest.raises(ValueError):
        load_controls(path, ["Other document" if change == "context" else "Policy"])


@title("Calibration runner records mismatches and errors before continuing")
def test_control_runner_preserves_mismatch_and_error_then_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    assert [r["status"] for r in results] == ["matched", "mismatch", "error"]
    assert original["response"] == "Original application answer"
    assert len(judges) == 3
    assert results[2]["error"]["type"] == "ValueError"


@title("Large calibration catalog loads without running every control")
def test_large_catalog_can_be_loaded_without_executing_all_cases(tmp_path: Path) -> None:
    controls = {
        "schema_version": 1,
        "required_context_fragments": ["Policy"],
        "cases": [{**_case(), "id": f"control-{i}"} for i in range(6)],
    }
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(controls))
    cases, checksum = load_controls(path, ["Policy"])
    assert len(cases) == 6
    assert len(checksum) == 64
    with pytest.raises(ValueError, match="maximum six judge calls"):
        select_controls(cases, None)
    chosen = select_controls(cases, ["control-4", "control-5"])
    assert [c["id"] for c in chosen] == ["control-4", "control-5"]


@title("Explicit oversized calibration batch is rejected before model calls")
def test_explicit_oversized_batch_is_rejected_before_model_calls() -> None:
    cases = [{**_case(), "id": f"control-{i}"} for i in range(4)]
    with pytest.raises(ValueError, match="maximum six judge calls"):
        select_controls(cases, [case["id"] for case in cases])


@title("Faithful but incomplete answer fails the required-fact check")
def test_faithful_incomplete_answer_still_fails_required_fact_check() -> None:
    response = Response()
    response.status_code = 200
    response._content = json.dumps(
        {
            "type": "textResponse",
            "error": None,
            "close": True,
            "textResponse": "Each employee receives 23 working days of paid leave per year.",
            "sources": [{"title": "policy.txt", "text": "23 working days; 12 calendar days"}],
        }
    ).encode()
    # A valid faithfulness score does not excuse omission of a requested fact.
    assertions.assert_quality_score(1.0, minimum=1.0)
    with pytest.raises(AssertionError, match="12 calendar days"):
        assertions.assert_rag_answer(
            response,
            fact_patterns={
                "23 working days": r"\b23\s+working\s+days\b",
                "12 calendar days": r"\b12\s+calendar\s+days\b",
            },
            document_title="policy.txt",
            source_fragments=("23 working days", "12 calendar days"),
        )
