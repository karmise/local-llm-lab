"""Scoped correctness evaluation inputs and prepared services."""

import json
from unittest.mock import Mock

import pytest
from requests import Response

from llm_testkit.config import Settings
from llm_testkit.evaluation.correctness import evaluate_correctness_report, score_correctness
from test_support.builders.correctness import (
        CorrectnessEvidenceScenario, CorrectnessQualityScenario, IncompleteControlScenario, _calls, _evidence, _result,
        _sample, append_judge_responses, make_failed_score_stub, make_incomplete_control, make_quality_scenario)
from test_support.builders.judge_scenarios import FailedServiceScenario, JudgeScenario, run_async
from test_support.builders.optional import load_ollama_judge
from test_support.data import common as case_data
from test_support.data.correctness import CASE, ROOT


@pytest.fixture
def failed_correctness_service(tmp_path, monkeypatch) -> FailedServiceScenario:
    pytest.importorskip("ragas")
    path = tmp_path / case_data.SAMPLE_FILE_NAME
    path.write_text(json.dumps(_sample()))
    transport = Mock()
    client = Mock()
    catalog = Response()
    catalog.status_code = 200
    catalog._content = json.dumps({
            "models": [{
            "name": case_data.TEST_MODEL,
            "digest": case_data.MODEL_DIGEST}]}).encode()
    client.list_models.return_value = catalog
    judge = Mock(calls=[{"error": "truncated"}], options={})
    monkeypatch.setattr("llm_testkit.evaluation.correctness.HttpClient", Mock(return_value=transport))
    monkeypatch.setattr("llm_testkit.evaluation.correctness.OllamaClient", Mock(return_value=client))
    monkeypatch.setattr("llm_testkit.evaluation.ollama_judge.OllamaJudge", Mock(return_value=judge))
    failed_score = make_failed_score_stub()
    monkeypatch.setattr("llm_testkit.evaluation.correctness.score_correctness", failed_score)
    return FailedServiceScenario(
            evaluate=lambda: evaluate_correctness_report(
            path, dataset_path=ROOT / "golden-policy.json", policy_file=ROOT / "company-policy.txt", case_id=CASE.id,
            settings=Settings(), judge_model=case_data.TEST_MODEL), transport=transport, judge=judge)


@pytest.fixture
def correctness_judge(rv, gv, expected) -> JudgeScenario:
    result = _result(rv, gv, expected)
    responses = []
    append_judge_responses([call["output"] for call in _calls(result)], responses)
    client = Mock()
    client.structured_chat.side_effect = responses
    judge = load_ollama_judge()(client, case_data.TEST_MODEL, max_calls=4)
    return JudgeScenario(
            evaluate=lambda: run_async(lambda: score_correctness(_sample(), judge)), judge=judge, client=client,
            expected_score=expected, maximum_calls=4)


@pytest.fixture
def correctness_evidence() -> CorrectnessEvidenceScenario:
    sample = _sample()
    return CorrectnessEvidenceScenario(sample, _evidence(sample))


@pytest.fixture
def correctness_quality(tmp_path) -> CorrectnessQualityScenario:
    return make_quality_scenario(tmp_path)


@pytest.fixture
def correctness_controls():
    return json.loads((ROOT / "correctness-controls.json").read_text())["cases"]


@pytest.fixture
def incomplete_control(correctness_controls) -> IncompleteControlScenario:
    return make_incomplete_control(correctness_controls)
