import asyncio
import copy
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.evaluation.faithfulness import load_sample, score_sample
from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.observation.evaluation_sample import build_sample

pytestmark = pytest.mark.unit


def _sample() -> dict:
    capture_id = "a" * 32
    capture = {
        "schema_version": 1, "boundary": "ollama-sdk-chat",
        "request": {"model": "test-model", "stream": False, "messages": [
            {"role": "system", "content": f"[LLM_TESTKIT_CAPTURE:{capture_id}]\n"
             "[CONTEXT 0]:\nEmployees receive 23 working days.\n[END CONTEXT 0]"},
            {"role": "user", "content": "How much leave?"},
        ]},
    }
    return build_sample(
        capture, question="How much leave?", answer="Employees receive 23 working days.",
        reference="23 working days.", expected_model="test-model", capture_id=capture_id,
    )


def test_sample_loader_rejects_context_substitution(tmp_path: Path) -> None:
    sample = _sample()
    sample["retrieved_contexts"] = ["An unrelated policy."]
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(sample))
    with pytest.raises(ValueError, match="contexts do not match"):
        load_sample(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "1"])
def test_quality_score_rejects_invalid_results(value: object) -> None:
    with pytest.raises(AssertionError):
        assertions.assert_quality_score(value)


def test_quality_threshold_rejects_low_score() -> None:
    with pytest.raises(AssertionError, match="below"):
        assertions.assert_quality_score(0.5, minimum=0.8)


def _judge_class():
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    pytest.importorskip("ragas", reason="Install the evaluation extra for RAGAS adapter checks")
    from llm_testkit.evaluation.ollama_judge import OllamaJudge
    return OllamaJudge


def _response(output: dict, *, done_reason: str = "stop") -> Response:
    response = Response()
    response.status_code = 200
    response._content = json.dumps({
        "model": "test-model", "done": True, "done_reason": done_reason,
        "message": {"content": json.dumps(output)},
    }).encode()
    return response


def test_real_ragas_pipeline_computes_supported_claim_ratio_without_network() -> None:
    Judge = _judge_class()
    client = Mock()
    client.structured_chat.side_effect = [
        _response({"statements": ["Supported claim.", "Unsupported claim."]}),
        _response({"statements": [
            {"statement": "Supported claim.", "reason": "Found in context.", "verdict": 1},
            {"statement": "Unsupported claim.", "reason": "Absent from context.", "verdict": 0},
        ]}),
    ]
    judge = Judge(client, "test-model")
    result = asyncio.run(score_sample(_sample(), judge))
    assertions.assert_field_equals(result, "value", 0.5)
    assertions.assert_field_equals({"calls": client.structured_chat.call_count}, "calls", 2)
    assertions.assert_field_equals(client.structured_chat.call_args.kwargs, "options", judge.options)
    with pytest.raises(ValueError, match="budget"):
        judge.generate("extra request", type(Mock()))


@pytest.mark.parametrize("output", [
    {"statements": []},
    {"statements": [{"statement": "Other claim.", "reason": "Wrong.", "verdict": 1}]},
    {"statements": [{"statement": "Claim.", "reason": "Wrong.", "verdict": 2}]},
])
def test_pipeline_rejects_missing_altered_or_invalid_verdicts(output: dict) -> None:
    Judge = _judge_class()
    client = Mock()
    client.structured_chat.side_effect = [
        _response({"statements": ["Claim."]}), _response(copy.deepcopy(output)),
    ]
    with pytest.raises((ValueError, AssertionError)):
        asyncio.run(score_sample(_sample(), Judge(client, "test-model")))


def test_judge_rejects_truncated_generation_without_retry() -> None:
    Judge = _judge_class()
    from pydantic import BaseModel

    class Statements(BaseModel):
        statements: list[str]

    client = Mock()
    client.structured_chat.return_value = _response({"statements": ["Claim."]}, done_reason="length")
    with pytest.raises(ValueError, match="truncated"):
        Judge(client, "test-model").generate("prompt", Statements)
    assertions.assert_field_equals({"calls": client.structured_chat.call_count}, "calls", 1)


def test_native_judge_request_disables_thinking_and_passes_schema() -> None:
    http = Mock()
    schema = {"type": "object"}
    OllamaClient(http).structured_chat(
        model="test-model", prompt="Judge this", schema=schema, options={"temperature": 0},
    )
    payload = http.request.call_args.kwargs["json"]
    assertions.assert_field_equals(payload, "think", False)
    assertions.assert_field_equals(payload, "stream", False)
    assertions.assert_field_equals(payload, "format", schema)


def test_cli_preserves_failure_as_error_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _judge_class()
    from llm_testkit.evaluation.faithfulness import main
    source = tmp_path / "sample.json"
    source.write_text("{}")
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["faithfulness", str(source), "--output", str(output)])
    assertions.assert_field_equals({"exit_code": main()}, "exit_code", 1)
    report = json.loads(output.read_text())
    assertions.assert_field_equals(report, "status", "error")
    assertions.assert_field_equals(report["error"], "type", "KeyError")


def test_cli_refuses_to_overwrite_report_before_model_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llm_testkit.evaluation.faithfulness import main
    output = tmp_path / "report.json"
    output.write_text("existing")
    monkeypatch.setattr("sys.argv", ["faithfulness", "missing-sample", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assertions.assert_field_equals({"exit_code": error.value.code}, "exit_code", 2)
    assertions.assert_field_equals({"contents": output.read_text()}, "contents", "existing")


def test_evaluation_service_preserves_judge_failure_and_closes_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _judge_class()
    from llm_testkit.config import Settings
    from llm_testkit.evaluation.faithfulness import evaluate_sample_report
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(_sample()))
    transport = Mock()
    client = Mock()
    catalog = Response()
    catalog.status_code = 200
    catalog._content = json.dumps({"models": [{"name": "test-model", "digest": "digest"}]}).encode()
    client.list_models.return_value = catalog
    judge = Mock(calls=[{"error_evidence": "incomplete generation"}], options={})
    monkeypatch.setattr("llm_testkit.evaluation.faithfulness.HttpClient", Mock(return_value=transport))
    monkeypatch.setattr("llm_testkit.evaluation.faithfulness.OllamaClient", Mock(return_value=client))
    monkeypatch.setattr("llm_testkit.evaluation.ollama_judge.OllamaJudge", Mock(return_value=judge))

    async def failed_score(sample: dict, judge: object) -> dict:
        raise ValueError("Judge generation truncated")

    monkeypatch.setattr("llm_testkit.evaluation.faithfulness.score_sample", failed_score)
    report = evaluate_sample_report(path, settings=Settings(), judge_model="test-model")
    assertions.assert_field_equals(report, "status", "error")
    assertions.assert_field_equals(report["error"], "type", "ValueError")
    assertions.assert_field_equals(report, "judge_calls", judge.calls)
    assertions.assert_field_equals({"transport_closes": transport.close.call_count}, "transport_closes", 1)
