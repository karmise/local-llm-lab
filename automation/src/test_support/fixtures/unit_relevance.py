"""Scoped relevance evaluation inputs and prepared services."""

import json
from unittest.mock import Mock

import pytest
from requests import Response

from llm_testkit.config import Settings
from llm_testkit.evaluation.relevance import (
    evaluate_relevance_report,
    score_relevance,
)
from llm_testkit.observation.evaluation_sample import build_sample
from test_support.builders.judge_scenarios import (
    FailedServiceScenario,
    RelevanceScenario,
    run_async,
)
from test_support.builders.relevance import (
    make_failed_score_stub,
    make_Judge_schema,
    result,
)
from test_support.data import common as case_data
from test_support.data.relevance import (
    CASE,
    ROOT,
)


@pytest.fixture
def failed_relevance_service(tmp_path, monkeypatch) -> FailedServiceScenario:
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
    path = tmp_path / case_data.SAMPLE_FILE_NAME
    path.write_text(json.dumps(sample))
    transport, client = Mock(), Mock()
    response = Response()
    response.status_code = 200
    response._content = json.dumps(
        {"models": [{"name": "test", "digest": case_data.MODEL_DIGEST}]}
    ).encode()
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
    return FailedServiceScenario(
        evaluate=lambda: evaluate_relevance_report(
            path,
            dataset_path=ROOT / "golden-policy.json",
            policy_file=ROOT / "company-policy.txt",
            case_id=CASE.id,
            settings=Settings(),
            judge_model="test",
        ),
        transport=transport,
        precision=precision,
        recall=recall,
    )


@pytest.fixture
def relevance_judges() -> RelevanceScenario:
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
    return RelevanceScenario(
        evaluate=lambda: run_async(lambda: score_relevance(sample, CASE, precision, recall)),
        precision=precision,
        recall=recall,
        expected=expected,
    )
