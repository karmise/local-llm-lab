import asyncio

import pytest

from llm_testkit.evaluation.relevance import (
    check_relevance_evidence,
    score_relevance,
    validate_relevance,
)
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import judges as judge_checks
from test_support.assertions import values as value_checks
from test_support.builders.relevance import (
    make_relevance_evidence,
    prepare_evidence_binding_case,
    prepare_invalid_relevance_case,
    result,
)
from test_support.data.relevance import (
    CASE,
    DATASET,
    EVIDENCE_BINDING_CHANGE_CASES,
    INVALID_RELEVANCE_CHANGE_CASES,
)
from test_support.fixtures.unit_relevance import (
    failed_relevance_service as failed_relevance_service,
)
from test_support.fixtures.unit_relevance import relevance_judges as relevance_judges

pytestmark = pytest.mark.unit


@title("Real RAGAS context metrics preserve retrieval order and detect missing reference facts")
def test_real_ragas_metrics_with_mocked_judge(relevance_judges):
    scored = relevance_judges.evaluate()
    judge_checks.relevance_matches(relevance_judges, scored)


@pytest.mark.parametrize("change", INVALID_RELEVANCE_CHANGE_CASES)
@title("Context metrics reject invalid verdicts and incomplete reference coverage [{param_id}]")
def test_invalid_relevance(change):
    data = result()
    prepare_invalid_relevance_case(change, data)
    errors.rejects(lambda: validate_relevance(data, CASE, 3), expected=ValueError)


@pytest.mark.parametrize("change", EVIDENCE_BINDING_CHANGE_CASES)
@title("Context evidence is bound to the sample, golden dataset and raw calls [{param_id}]")
def test_evidence_binding(change):
    sample, evidence, _ = make_relevance_evidence()
    prepare_evidence_binding_case(change, evidence)
    errors.rejects(
        lambda: check_relevance_evidence(evidence, "sample", sample, DATASET), expected=ValueError
    )


@title("Context evaluation rejects oversized retrieval without silently truncating or judging")
def test_context_budget(mock_factory):
    pytest.importorskip("ragas")
    judge = mock_factory()
    errors.rejects(
        lambda: asyncio.run(
            score_relevance({"retrieved_contexts": ["Context"] * 5}, CASE, judge, judge)
        ),
        expected=ValueError,
        match="never truncated",
    )
    value_checks.falsy(judge.mock_calls)


@title("Context evaluator retains raw failed calls and closes its transport")
def test_service_error_evidence(failed_relevance_service):
    report = failed_relevance_service.evaluate()
    judge_checks.failure_evidence_is_retained(failed_relevance_service, report)


@title("Completed relevance evidence retains validated original metric values")
def test_valid_evidence_binding():
    sample, evidence, expected = make_relevance_evidence()
    value_checks.equal(check_relevance_evidence(evidence, "sample", sample, DATASET), expected)
