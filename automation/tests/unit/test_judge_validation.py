import pytest

from llm_testkit.datasets.judge_controls import select_judge_controls
from llm_testkit.evaluation.judge_validation import label_mismatches, result_from_evidence, summarize_validation
from llm_testkit.reporting.steps import title
from test_support.assertions import errors
from test_support.assertions import judge_validation as checks
from test_support.assertions import values
from test_support.data.judge_validation import CONTROL_IDS, INVALID_CATALOG_CHANGES
from test_support.fixtures.unit_benchmark import benchmark_data as benchmark_data
from test_support.fixtures.unit_judge_validation import corrupt_precision_evidence as corrupt_precision_evidence
from test_support.fixtures.unit_judge_validation import failed_validation_batch as failed_validation_batch
from test_support.fixtures.unit_judge_validation import invalid_judge_catalog as invalid_judge_catalog
from test_support.fixtures.unit_judge_validation import judge_catalog as judge_catalog
from test_support.fixtures.unit_judge_validation import judge_cli as judge_cli
from test_support.fixtures.unit_judge_validation import judge_control as judge_control
from test_support.fixtures.unit_judge_validation import real_metric_control as real_metric_control
from test_support.fixtures.unit_judge_validation import reversed_recall_labels as reversed_recall_labels
from test_support.fixtures.unit_judge_validation import saved_judge_review as saved_judge_review

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("control_id", CONTROL_IDS)
@title("Real RAGAS metric agrees with labelled mocked judge evidence [{param_id}]")
def test_real_metric_preserves_control_labels(real_metric_control):
    results = real_metric_control()
    checks.matched_control(results)


@pytest.mark.parametrize("change", INVALID_CATALOG_CHANGES)
@title("Judge controls reject stale sources and invalid labels before model calls [{param_id}]")
def test_invalid_catalog_is_rejected(invalid_judge_catalog):
    errors.rejects(invalid_judge_catalog, expected=ValueError)


@title("Judge validation rejects an insufficient budget before model calls")
def test_budget_is_checked_before_execution(judge_catalog):
    errors.rejects(lambda: select_judge_controls(judge_catalog, None, 17), expected=ValueError, match="budget")


@title("Judge validation rejects unknown control selection")
def test_unknown_selection_is_rejected(judge_catalog):
    errors.rejects(lambda: select_judge_controls(judge_catalog, ["unknown"], 18), expected=ValueError, match="known")


@title("Judge validation preserves mismatches and errors while checking remaining controls")
def test_independent_failures_remain_visible(failed_validation_batch):
    results = failed_validation_batch.run()
    checks.failures_are_retained(results)
    checks.experimental_decision(summarize_validation(results))


@title("A matching recall score cannot hide reversed fact labels")
def test_equal_score_does_not_hide_wrong_claim_verdicts(reversed_recall_labels):
    control, result = reversed_recall_labels
    values.length(label_mismatches(result, control), 2)


@title("Precision score must agree with ordered raw judge verdicts")
def test_forged_score_is_rejected(corrupt_precision_evidence):
    control, calls = corrupt_precision_evidence
    errors.rejects(lambda: result_from_evidence(control, calls, 0.5), expected=ValueError, match="raw verdicts")


@title("An empty judge experiment cannot pass")
def test_empty_experiment_is_rejected():
    errors.rejects(lambda: summarize_validation([]), expected=ValueError, match="nonempty")


@title("Saved benchmark review preserves pending approval and original outcomes")
def test_saved_review_does_not_approve_results(saved_judge_review):
    report = saved_judge_review.review()
    checks.unapproved_review(report)


@title("Saved benchmark review rejects corrupted evidence")
def test_saved_review_validates_original_artifacts(saved_judge_review):
    saved_judge_review.corrupt_evidence()
    errors.rejects(saved_judge_review.review, expected=ValueError, match="checksum mismatch")


@title("Judge validation preview performs no network operations")
def test_preview_has_no_model_calls(judge_cli):
    invoke, transport, _ = judge_cli
    values.equal(invoke(["--dry-run"]), 0)
    values.equal(transport.call_count, 0)


@title("Judge validation refuses overwriting saved evidence before network access")
def test_cli_preserves_existing_evidence(judge_cli):
    invoke, transport, output = judge_cli
    error = errors.rejects(lambda: invoke(["--output", str(output)]), expected=SystemExit)
    values.equal(error.value.code, 2)
    values.equal(output.read_text(), "original")
    values.equal(transport.call_count, 0)
