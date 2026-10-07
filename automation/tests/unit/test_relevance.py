import asyncio
import json
from unittest.mock import Mock

import pytest
from requests import Response

from llm_testkit.config import Settings
from llm_testkit.evaluation.relevance import (
    check_relevance_evidence,
    evaluate_relevance_report,
    score_relevance,
    validate_relevance,
)
from llm_testkit.observation.evaluation_sample import build_sample
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import values as value_checks
from test_support.builders.relevance import (
    make_failed_score_stub,
    make_Judge_schema,
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
    ROOT,
)

pytestmark = pytest.mark.unit


@title("Real RAGAS context metrics preserve retrieval order and detect missing reference facts")
def test_real_ragas_metrics_with_mocked_judge():
    pytest.importorskip("ragas")

    Judge = make_Judge_schema()

    expected = result()
    precision = Judge(expected["precision_verdicts"])
    recall = Judge([{"classifications": expected["recall_classifications"]}])
    sample = {
        "user_input": CASE.question,
        "reference": CASE.reference,
        "retrieved_contexts": ["Relevant", "Unrelated", "Relevant"],
    }
    value_checks.equal(asyncio.run(score_relevance(sample, CASE, precision, recall)), expected)
    value_checks.length(precision.calls, 3)
    value_checks.length(recall.calls, 1)
    value_checks.equal(expected["context_precision"], pytest.approx(5 / 6))


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
def test_context_budget():
    pytest.importorskip("ragas")
    judge = Mock()
    errors.rejects(
        lambda: asyncio.run(
            score_relevance({"retrieved_contexts": ["Context"] * 5}, CASE, judge, judge)
        ),
        expected=ValueError,
        match="never truncated",
    )
    value_checks.falsy(judge.mock_calls)


@title("Context evaluator retains raw failed calls and closes its transport")
def test_service_error_evidence(tmp_path, monkeypatch):

    pytest.importorskip("ragas")
    identifier = "a" * 32
    capture = {
        "schema_version": 1,
        "boundary": "ollama-sdk-chat",
        "request": {
            "model": "test",
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": f"[LLM_TESTKIT_CAPTURE:{identifier}]\n[CONTEXT 0]:\nPolicy\n[END CONTEXT 0]",
                },
                {"role": "user", "content": CASE.question},
            ],
        },
    }
    sample = build_sample(
        capture,
        question=CASE.question,
        answer=CASE.reference,
        reference=CASE.reference,
        expected_model="test",
        capture_id=identifier,
    )
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(sample))
    transport, client = Mock(), Mock()
    response = Response()
    response.status_code = 200
    response._content = json.dumps({"models": [{"name": "test", "digest": "digest"}]}).encode()
    client.list_models.return_value = response
    precision = Mock(calls=[{"error": "Truncated"}], options={})
    recall = Mock(calls=[], options={})
    monkeypatch.setattr("llm_testkit.evaluation.relevance.HttpClient", Mock(return_value=transport))
    monkeypatch.setattr("llm_testkit.evaluation.relevance.OllamaClient", Mock(return_value=client))
    monkeypatch.setattr(
        "llm_testkit.evaluation.ollama_judge.OllamaJudge", Mock(side_effect=[precision, recall])
    )

    failed_score = make_failed_score_stub()

    monkeypatch.setattr("llm_testkit.evaluation.relevance.score_relevance", failed_score)
    report = evaluate_relevance_report(
        path,
        dataset_path=ROOT / "golden-policy.json",
        policy_file=ROOT / "company-policy.txt",
        case_id=CASE.id,
        settings=Settings(),
        judge_model="test",
    )
    value_checks.equal(report["status"], "error")
    value_checks.equal(report["precision_calls"], precision.calls)
    value_checks.equal(report["recall_calls"], [])
    value_checks.equal(transport.close.call_count, 1)
    value_checks.equal(report["judge_configuration"]["maximum_calls"], 2)


@title("Completed relevance evidence retains validated original metric values")
def test_valid_evidence_binding():
    sample, evidence, expected = make_relevance_evidence()
    value_checks.equal(check_relevance_evidence(evidence, "sample", sample, DATASET), expected)
