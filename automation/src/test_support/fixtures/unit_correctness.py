"""Scoped correctness evaluation inputs and prepared services."""

import json
from unittest.mock import Mock

import pytest
from requests import Response

from llm_testkit.config import Settings
from llm_testkit.evaluation.correctness import (
    evaluate_correctness_report,
    score_correctness,
)
from test_support.builders.correctness import (
    _calls,
    _result,
    _sample,
    append_judge_responses,
    make_failed_score_stub,
)
from test_support.builders.judge_scenarios import FailedServiceScenario, JudgeScenario, run_async
from test_support.builders.optional import load_ollama_judge
from test_support.data.correctness import (
    CASE,
    ROOT,
)


@pytest.fixture
def failed_correctness_service(tmp_path, monkeypatch) -> FailedServiceScenario:
    pytest.importorskip("ragas")
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(_sample()))
    transport = Mock()
    client = Mock()
    catalog = Response()
    catalog.status_code = 200
    catalog._content = json.dumps({"models": [{"name": "test-model", "digest": "digest"}]}).encode()
    client.list_models.return_value = catalog
    judge = Mock(calls=[{"error": "truncated"}], options={})
    monkeypatch.setattr(
        "llm_testkit.evaluation.correctness.HttpClient", Mock(return_value=transport)
    )
    monkeypatch.setattr(
        "llm_testkit.evaluation.correctness.OllamaClient", Mock(return_value=client)
    )
    monkeypatch.setattr("llm_testkit.evaluation.ollama_judge.OllamaJudge", Mock(return_value=judge))
    failed_score = make_failed_score_stub()
    monkeypatch.setattr("llm_testkit.evaluation.correctness.score_correctness", failed_score)
    return FailedServiceScenario(
        evaluate=lambda: evaluate_correctness_report(
            path,
            dataset_path=ROOT / "golden-policy.json",
            policy_file=ROOT / "company-policy.txt",
            case_id=CASE.id,
            settings=Settings(),
            judge_model="test-model",
        ),
        transport=transport,
        judge=judge,
    )


@pytest.fixture
def correctness_judge(rv, gv, expected) -> JudgeScenario:
    result = _result(rv, gv, expected)
    responses = []
    append_judge_responses([call["output"] for call in _calls(result)], responses)
    client = Mock()
    client.structured_chat.side_effect = responses
    judge = load_ollama_judge()(client, "test-model", max_calls=4)
    return JudgeScenario(
        evaluate=lambda: run_async(lambda: score_correctness(_sample(), judge)),
        judge=judge,
        client=client,
        expected_score=expected,
        maximum_calls=4,
    )
