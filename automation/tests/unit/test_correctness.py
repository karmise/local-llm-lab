import pytest

from llm_testkit.evaluation.correctness import (
    bind_case,
    check_control,
    main,
    validate_result,
)
from llm_testkit.reporting.steps import title
from test_support.assertions import correctness as correctness_checks
from test_support.assertions import errors as errors
from test_support.assertions import judges as judge_checks
from test_support.assertions import values as value_checks
from test_support.builders.correctness import (
    _result,
    _sample,
    prepare_invalid_claim_evidence_is_rejected_case,
)
from test_support.data import common as case_data
from test_support.data.correctness import (
    CASE,
    DATASET,
    INVALID_CLAIM_EVIDENCE_IS_REJECTED_CHANGE_CASES,
    REAL_RAGAS_CORRECTNESS_WITH_MOCKED_JUDGE_RV_GV_EXPECTED_CASES,
    REAL_RAGAS_CORRECTNESS_WITH_MOCKED_JUDGE_RV_GV_EXPECTED_IDS,
    STALE_DATASET_METADATA,
    UNRELATED_OR_SYNTHETIC_EVIDENCE_IS_REJECTED_CHANGE_CASES,
)
from test_support.fixtures.unit_correctness import correctness_controls as correctness_controls
from test_support.fixtures.unit_correctness import correctness_evidence as correctness_evidence
from test_support.fixtures.unit_correctness import correctness_judge as correctness_judge
from test_support.fixtures.unit_correctness import correctness_quality as correctness_quality
from test_support.fixtures.unit_correctness import (
    failed_correctness_service as failed_correctness_service,
)
from test_support.fixtures.unit_correctness import incomplete_control as incomplete_control

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("rv", "gv", "expected"),
    REAL_RAGAS_CORRECTNESS_WITH_MOCKED_JUDGE_RV_GV_EXPECTED_CASES,
    ids=REAL_RAGAS_CORRECTNESS_WITH_MOCKED_JUDGE_RV_GV_EXPECTED_IDS,
)
@title(
    "Real RAGAS factual F1 distinguishes correct, incomplete and incorrect evidence [{param_id}]"
)
def test_real_ragas_correctness_with_mocked_judge(rv, gv, expected, correctness_judge):
    scored = correctness_judge.evaluate()
    judge_checks.score_matches(correctness_judge, scored)
    judge_checks.budget_is_exhausted(correctness_judge)


@pytest.mark.parametrize("change", INVALID_CLAIM_EVIDENCE_IS_REJECTED_CHANGE_CASES)
@title("Correctness rejects incomplete, duplicated or inconsistent judge evidence [{param_id}]")
def test_invalid_claim_evidence_is_rejected(change):
    result = _result()
    prepare_invalid_claim_evidence_is_rejected_case(change, result)
    errors.rejects(lambda: validate_result(result), expected=ValueError)


@pytest.mark.parametrize("change", UNRELATED_OR_SYNTHETIC_EVIDENCE_IS_REJECTED_CHANGE_CASES)
@title(
    "Correctness evidence remains bound to application sample and golden expectations [{param_id}]"
)
def test_unrelated_or_synthetic_evidence_is_rejected(change, correctness_evidence):
    correctness_evidence.invalidate(change)
    correctness_checks.rejects_unbound_evidence(correctness_evidence)


@title("Correctness rejects substituted reference and stale sample provenance")
def test_sample_must_use_current_golden_reference():
    sample = _sample()
    sample["reference"] = "Edited reference"
    errors.rejects(
        lambda: bind_case(sample, DATASET, CASE.id), expected=ValueError, match="question/reference"
    )
    sample = _sample()
    sample["metadata"] = case_data.fresh(STALE_DATASET_METADATA)
    errors.rejects(
        lambda: bind_case(sample, DATASET, CASE.id), expected=ValueError, match="provenance"
    )


@title("Optional correctness records a valid low score and independently rejects broken evidence")
def test_quality_report_adds_independent_correctness_measurement(correctness_quality):
    report = correctness_quality.build_report()
    correctness_checks.low_score_remains_a_measurement(report)
    correctness_quality.invalidate_sample_checksum()
    report = correctness_quality.build_report()
    correctness_checks.invalid_correctness_does_not_invalidate_faithfulness(report)


@title("Correctness CLI refuses overwriting evidence before model calls")
def test_correctness_cli_refuses_overwrite(tmp_path, monkeypatch):

    output = tmp_path / "existing.json"
    output.write_text("existing")
    monkeypatch.setattr(
        "sys.argv", ["correctness", "sample", "--case", "paid_leave", "--output", str(output)]
    )
    error = errors.rejects(lambda: main(), expected=SystemExit)
    value_checks.equal(error.value.code, 2)
    value_checks.equal(output.read_text(), "existing")


@title("Correctness controls tolerate decomposition variation while verifying omission labels")
def test_incomplete_control_checks_semantics_instead_of_fixed_claim_count(incomplete_control):
    check_control(incomplete_control.result, incomplete_control.control)
    incomplete_control.attribute_missing_claim()
    correctness_checks.rejects_incorrect_control_labels(incomplete_control)


@title("All curated correctness controls have valid labelled expectations")
def test_control_catalog_has_valid_expectations(correctness_controls):
    correctness_checks.check_control_expectations(correctness_controls)


@title("Correctness report rejects summaries that differ from raw judge evidence")
def test_raw_calls_must_match_summary(correctness_evidence):
    correctness_evidence.corrupt_raw_claims()
    correctness_checks.rejects_unbound_evidence(correctness_evidence)


@title("Correctness service preserves failure evidence and closes its transport")
def test_correctness_service_closes_transport_on_judge_error(failed_correctness_service):
    report = failed_correctness_service.evaluate()
    judge_checks.failure_evidence_is_retained(failed_correctness_service, report)
