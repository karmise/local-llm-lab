import asyncio
import json
from pathlib import Path

import pytest

from llm_testkit import assertions
from llm_testkit.evaluation.calibration import evaluate_controls, load_controls, select_controls
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import values as value_checks
from test_support.assertions.calibration import check_invalid_control_catalog
from test_support.builders.calibration import (
    _case,
    _result,
    make_alternative_control,
    make_control_catalog,
    make_error_control,
    make_factory_stub,
    make_fake_score_stub,
    make_incomplete_policy_reply,
    make_large_control_catalog,
    make_mismatched_control,
    make_numbered_control,
    prepare_control_loader_rejects_wrong_context_or_bad_labels_case,
)
from test_support.data import common as case_data
from test_support.data.calibration import (
    CONTROL_LOADER_REJECTS_WRONG_CONTEXT_OR_BAD_LABELS_CHANGE_CASES,
    FACTS_23_WORKING_DAYS_12_CALENDAR_DAYS_INPUT,
    ORIGINAL_INPUT,
    RUNNER_REJECTS_EMPTY_OR_EXCESSIVE_CONTROL_RUNS_COUNT_CASES,
)

pytestmark = pytest.mark.unit


@title("Unknown calibration control is rejected before model calls")
def test_unknown_control_selection_fails_before_model_calls() -> None:
    errors.rejects(
        lambda: select_controls([_case()], ["typo"]), expected=ValueError, match="Unknown controls"
    )


@title("Targeted calibration selection removes duplicate controls")
def test_targeted_selection_deduplicates_controls() -> None:
    selected = select_controls([_case(), make_alternative_control()], ["mixed", "mixed"])
    value_checks.equal([case["id"] for case in selected], ["mixed"])


@pytest.mark.parametrize("count", RUNNER_REJECTS_EMPTY_OR_EXCESSIVE_CONTROL_RUNS_COUNT_CASES)
@title("Calibration runner rejects empty or excessive control batches [{param_id}]")
def test_runner_rejects_empty_or_excessive_control_runs(count: int, mock_factory) -> None:
    factory = mock_factory()
    errors.rejects(
        lambda: asyncio.run(
            evaluate_controls(case_data.fresh(case_data.EMPTY_OBJECT), [_case()] * count, factory)
        ),
        expected=ValueError,
        match="between one and three",
    )
    value_checks.equal(factory.call_count, 0)


@title("Calibration rejects reversed claim verdicts even when the score matches")
def test_same_score_with_reversed_verdicts_fails_control_check() -> None:
    result = _result()
    result["verdicts"][0]["verdict"] = 0
    result["verdicts"][1]["verdict"] = 1
    errors.rejects(
        lambda: assertions.assert_calibration_result(
            result, expected_score=0.5, claims=_case()["claims"]
        ),
        expected=AssertionError,
    )


@title("Calibration rejects a missing claim even when the score matches")
def test_missing_claim_fails_even_when_score_matches() -> None:
    result = _result()
    result["verdicts"].pop()
    errors.rejects(
        lambda: assertions.assert_calibration_result(
            result, expected_score=0.5, claims=_case()["claims"]
        ),
        expected=AssertionError,
        match="number of claims",
    )


@title("Two expected calibration claims cannot share one combined judge statement")
def test_two_expected_claims_cannot_match_one_combined_statement() -> None:
    result = _result()
    result["verdicts"][0]["statement"] = "Leave 23; gym 5000."
    result["verdicts"][1]["statement"] = "Other claim."
    errors.rejects(
        lambda: assertions.assert_calibration_result(
            result, expected_score=0.5, claims=_case()["claims"]
        ),
        expected=AssertionError,
        match="distinct",
    )


@pytest.mark.parametrize("change", CONTROL_LOADER_REJECTS_WRONG_CONTEXT_OR_BAD_LABELS_CHANGE_CASES)
@title("Calibration loader rejects unrelated context and invalid control labels [{param_id}]")
def test_control_loader_rejects_wrong_context_or_bad_labels(tmp_path: Path, change: str) -> None:
    controls = make_control_catalog()
    prepare_control_loader_rejects_wrong_context_or_bad_labels_case(change, controls)
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(controls))
    check_invalid_control_catalog(change, path)


@title("Calibration runner records mismatches and errors before continuing")
def test_control_runner_preserves_mismatch_and_error_then_continues(
        monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    cases = [_case(), make_mismatched_control(), make_error_control()]
    original = case_data.fresh(ORIGINAL_INPUT)
    observed_responses = []

    fake_score = make_fake_score_stub(observed_responses)

    monkeypatch.setattr("llm_testkit.evaluation.calibration.score_sample", fake_score)
    judges = []

    factory = make_factory_stub(judges)

    results = asyncio.run(evaluate_controls(original, cases, factory))
    value_checks.equal([r["status"] for r in results], ["matched", "mismatch", "error"])
    value_checks.equal(original["response"], "Original application answer")
    value_checks.length(judges, 3)
    value_checks.equal(results[2]["error"]["type"], "ValueError")


@title("Large calibration catalog loads without running every control")
def test_large_catalog_can_be_loaded_without_executing_all_cases(tmp_path: Path) -> None:
    controls = make_large_control_catalog()
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(controls))
    cases, checksum = load_controls(path, ["Policy"])
    value_checks.length(cases, 6)
    value_checks.length(checksum, 64)
    errors.rejects(
        lambda: select_controls(cases, None), expected=ValueError, match="maximum six judge calls"
    )
    chosen = select_controls(cases, ["control-4", "control-5"])
    value_checks.equal([c["id"] for c in chosen], ["control-4", "control-5"])


@title("Explicit oversized calibration batch is rejected before model calls")
def test_explicit_oversized_batch_is_rejected_before_model_calls() -> None:
    cases = [make_numbered_control(i) for i in range(4)]
    errors.rejects(
        lambda: select_controls(cases, [case["id"] for case in cases]),
        expected=ValueError,
        match="maximum six judge calls",
    )


@title("Faithful but incomplete answer fails the required-fact check")
def test_faithful_incomplete_answer_still_fails_required_fact_check(response_factory) -> None:
    response = response_factory()
    response.status_code = 200
    response._content = json.dumps(make_incomplete_policy_reply()).encode()
    # A valid faithfulness score does not excuse omission of a requested fact.
    assertions.assert_quality_score(1.0, minimum=1.0)
    errors.rejects(
        lambda: assertions.assert_rag_answer(
            response,
            fact_patterns=case_data.fresh(FACTS_23_WORKING_DAYS_12_CALENDAR_DAYS_INPUT),
            document_title=case_data.POLICY_DOCUMENT_TITLE,
            source_fragments=("23 working days", "12 calendar days"),
        ),
        expected=AssertionError,
        match="12 calendar days",
    )
