import asyncio
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.evaluation.faithfulness import load_sample, score_sample
from llm_testkit.reporting.steps import title
from test_support.builders.evaluation import _judge_class, _response, _sample

pytestmark = pytest.mark.unit


@title("Evaluation sample loader rejects substituted model context")
def test_sample_loader_rejects_context_substitution(tmp_path: Path) -> None:
    sample = _sample()
    sample["retrieved_contexts"] = ["An unrelated policy."]
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(sample))
    with pytest.raises(ValueError, match="contexts do not match"):
        load_sample(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "1"])
@title("Quality-score check rejects invalid values [{param_id}]")
def test_quality_score_rejects_invalid_results(value: object) -> None:
    with pytest.raises(AssertionError):
        assertions.assert_quality_score(value)


@title("Quality threshold rejects a score below the required minimum")
def test_quality_threshold_rejects_low_score() -> None:
    with pytest.raises(AssertionError, match="below"):
        assertions.assert_quality_score(0.5, minimum=0.8)


@title("RAGAS computes supported-claim ratio from mocked judge responses")
def test_real_ragas_pipeline_computes_supported_claim_ratio_without_network() -> None:
    Judge = _judge_class()
    client = Mock()
    client.structured_chat.side_effect = [
        _response({"statements": ["Supported claim.", "Unsupported claim."]}),
        _response(
            {
                "statements": [
                    {"statement": "Supported claim.", "reason": "Found in context.", "verdict": 1},
                    {
                        "statement": "Unsupported claim.",
                        "reason": "Absent from context.",
                        "verdict": 0,
                    },
                ]
            }
        ),
    ]
    judge = Judge(client, "test-model")
    result = asyncio.run(score_sample(_sample(), judge))
    assert result["value"] == 0.5
    assert client.structured_chat.call_count == 2
    assert client.structured_chat.call_args.kwargs["options"] == judge.options
    with pytest.raises(ValueError, match="budget"):
        judge.generate("extra request", type(Mock()))


@pytest.mark.parametrize(
    "output",
    [
        {"statements": []},
        {"statements": [{"statement": "Other claim.", "reason": "Wrong.", "verdict": 1}]},
        {"statements": [{"statement": "Claim.", "reason": "Wrong.", "verdict": 2}]},
    ],
)
@title("Faithfulness pipeline rejects missing, altered or invalid claim verdicts [{param_id}]")
def test_pipeline_rejects_missing_altered_or_invalid_verdicts(output: dict) -> None:
    Judge = _judge_class()
    client = Mock()
    client.structured_chat.side_effect = [
        _response({"statements": ["Claim."]}),
        _response(copy.deepcopy(output)),
    ]
    with pytest.raises((ValueError, AssertionError)):
        asyncio.run(score_sample(_sample(), Judge(client, "test-model")))


@title("Local judge rejects truncated generation without retrying")
def test_judge_rejects_truncated_generation_without_retry() -> None:
    Judge = _judge_class()
    from pydantic import BaseModel

    class Statements(BaseModel):
        statements: list[str]

    client = Mock()
    client.structured_chat.return_value = _response(
        {"statements": ["Claim."]}, done_reason="length"
    )
    with pytest.raises(ValueError, match="truncated"):
        Judge(client, "test-model").generate("prompt", Statements)
    assert client.structured_chat.call_count == 1


@title("Native judge request disables thinking and includes the response schema")
def test_native_judge_request_disables_thinking_and_passes_schema() -> None:
    http = Mock()
    schema = {"type": "object"}
    OllamaClient(http).structured_chat(
        model="test-model",
        prompt="Judge this",
        schema=schema,
        options={"temperature": 0},
    )
    payload = http.request.call_args.kwargs["json"]
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["format"] == schema


@title("Faithfulness CLI preserves a failure in its error report")
def test_cli_preserves_failure_as_error_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _judge_class()
    from llm_testkit.evaluation.faithfulness import main

    source = tmp_path / "sample.json"
    source.write_text("{}")
    output = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["faithfulness", str(source), "--output", str(output)])
    assert main() == 1
    report = json.loads(output.read_text())
    assert report["status"] == "error"
    assert report["error"]["type"] == "KeyError"


@title("Faithfulness CLI refuses to overwrite an existing report before model calls")
def test_cli_refuses_to_overwrite_report_before_model_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_testkit.evaluation.faithfulness import main

    output = tmp_path / "report.json"
    output.write_text("existing")
    monkeypatch.setattr("sys.argv", ["faithfulness", "missing-sample", "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert output.read_text() == "existing"


@title("Evaluation service preserves judge failure and closes its HTTP transport")
def test_evaluation_service_preserves_judge_failure_and_closes_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setattr(
        "llm_testkit.evaluation.faithfulness.HttpClient", Mock(return_value=transport)
    )
    monkeypatch.setattr(
        "llm_testkit.evaluation.faithfulness.OllamaClient", Mock(return_value=client)
    )
    monkeypatch.setattr("llm_testkit.evaluation.ollama_judge.OllamaJudge", Mock(return_value=judge))

    async def failed_score(sample: dict, judge: object) -> dict:
        raise ValueError("Judge generation truncated")

    monkeypatch.setattr("llm_testkit.evaluation.faithfulness.score_sample", failed_score)
    report = evaluate_sample_report(path, settings=Settings(), judge_model="test-model")
    assert report["status"] == "error"
    assert report["error"]["type"] == "ValueError"
    assert report["judge_calls"] == judge.calls
    assert transport.close.call_count == 1


def test_live_faithfulness_rejects_score_inconsistent_with_verdicts(monkeypatch):
    collections = pytest.importorskip("ragas.metrics.collections")

    metric = Mock()
    from unittest.mock import AsyncMock

    metric.ascore = AsyncMock(return_value=Mock(value=1.0))
    monkeypatch.setattr(collections, "Faithfulness", Mock(return_value=metric))
    judge = Mock(
        calls=[
            {"output": {"statements": ["Claim."]}},
            {"output": {"statements": [{"statement": "Claim.", "verdict": 0}]}},
        ]
    )
    with pytest.raises(ValueError, match="score"):
        asyncio.run(score_sample(_sample(), judge))
