"""Scoped evaluation evaluation inputs and prepared services."""

import json
from copy import deepcopy
from unittest.mock import Mock

import pytest
from requests import Response

from llm_testkit.config import Settings
from llm_testkit.evaluation.faithfulness import (
    evaluate_sample_report,
    score_sample,
)
from test_support.builders.evaluation import (
    _judge_class,
    _response,
    _sample,
    make_failed_score_stub,
    make_Statements_schema,
)
from test_support.builders.judge_scenarios import FailedServiceScenario, JudgeScenario, run_async
from test_support.data.evaluation import CLAIM_EXTRACTION, SUPPORTED_CLAIM_OUTPUTS


@pytest.fixture
def failed_evaluation_service(tmp_path, monkeypatch) -> FailedServiceScenario:
    _judge_class()
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(_sample()))
    transport = Mock()
    client = Mock()
    catalog = Response()
    catalog.status_code = 200
    catalog._content = json.dumps({"models": [{"name": "test-model", "digest": "digest"}]}).encode()
    client.list_models.return_value = catalog
    judge = Mock(calls=[{"error_evidence": "incomplete generation"}], options={})
    monkeypatch.setattr(
        "llm_testkit.evaluation.faithfulness.HttpClient", Mock(return_value=transport)
    )
    monkeypatch.setattr(
        "llm_testkit.evaluation.faithfulness.OllamaClient", Mock(return_value=client)
    )
    monkeypatch.setattr("llm_testkit.evaluation.ollama_judge.OllamaJudge", Mock(return_value=judge))
    failed_score = make_failed_score_stub()
    monkeypatch.setattr("llm_testkit.evaluation.faithfulness.score_sample", failed_score)
    return FailedServiceScenario(
        evaluate=lambda: evaluate_sample_report(
            path, settings=Settings(), judge_model="test-model"
        ),
        transport=transport,
        judge=judge,
    )


@pytest.fixture
def faithfulness_judge() -> JudgeScenario:
    client = Mock()
    client.structured_chat.side_effect = [
        _response(output) for output in deepcopy(SUPPORTED_CLAIM_OUTPUTS)
    ]
    judge = _judge_class()(client, "test-model")
    return JudgeScenario(
        evaluate=lambda: run_async(lambda: score_sample(_sample(), judge)),
        judge=judge,
        client=client,
        expected_score=0.5,
        maximum_calls=2,
        check_options=True,
    )


@pytest.fixture
def invalid_faithfulness_judge(output) -> JudgeScenario:
    client = Mock()
    client.structured_chat.side_effect = [_response(CLAIM_EXTRACTION), _response(deepcopy(output))]
    judge = _judge_class()(client, "test-model")
    return JudgeScenario(
        evaluate=lambda: run_async(lambda: score_sample(_sample(), judge)),
        judge=judge,
        client=client,
        expected_score=0.0,
        maximum_calls=2,
    )


@pytest.fixture
def truncated_judge() -> JudgeScenario:
    client = Mock()
    client.structured_chat.return_value = _response(CLAIM_EXTRACTION, done_reason="length")
    judge = _judge_class()(client, "test-model")
    schema = make_Statements_schema()
    return JudgeScenario(
        evaluate=lambda: judge.generate("prompt", schema),
        judge=judge,
        client=client,
        expected_score=0.0,
        maximum_calls=1,
    )
