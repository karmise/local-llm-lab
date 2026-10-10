"""Faithfulness: RAGAS scoring through the local judge, saved-sample integrity, the report service and the CLI."""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock

import pytest

from llm_testkit.config import Settings
from llm_testkit.evaluation.faithfulness import evaluate_sample_report, load_sample, main, score_sample, validate_result
from llm_testkit.reporting.steps import title
from test_support.builders.faithfulness import make_result, make_sample
from test_support.builders.ollama import chat_response, model_catalog
from test_support.builders.optional import load_ollama_judge
from test_support.data.common import MODEL_DIGEST, TEST_MODEL

pytestmark = pytest.mark.unit

SCORE = "llm_testkit.evaluation.faithfulness.score_sample"
EXTRACTED = {"statements": ["Supported claim.", "Unsupported claim."]}
VERIFIED = {
        "statements": [{
        "statement": "Supported claim.",
        "reason": "Found in context.",
        "verdict": 1}, {
        "statement": "Unsupported claim.",
        "reason": "Absent from context.",
        "verdict": 0}]}


def scripted_judge(*outputs: dict):
    """A real OllamaJudge with the default two-call budget whose transport replays the given outputs."""
    client = Mock()
    client.structured_chat.side_effect = [chat_response(output) for output in outputs]
    return load_ollama_judge()(client, TEST_MODEL), client


@title("RAGAS faithfulness is the share of extracted claims the judge finds supported")
def test_ragas_faithfulness_is_share_of_supported_claims():
    judge, client = scripted_judge(EXTRACTED, VERIFIED)

    result = asyncio.run(score_sample(make_sample(), judge))

    assert result["value"] == 0.5
    assert result["statements"] == EXTRACTED["statements"]
    assert [verdict["verdict"] for verdict in result["verdicts"]] == [1, 0]
    assert client.structured_chat.call_count == 2
    assert client.structured_chat.call_args.kwargs["options"] == judge.options
    with pytest.raises(ValueError, match="budget"):
        judge.generate("An additional request must be rejected", object)
    assert client.structured_chat.call_count == 2, "Budget rejection must happen before another transport call"


@pytest.mark.parametrize(
        ("verdicts", "error", "message"),
        [
        # RAGAS turns an empty verdict list into a NaN score, which the quality-score check rejects first.
        pytest.param({"statements": []}, AssertionError, "finite quality score", id="no-verdicts"),
        pytest.param({"statements": [{
        "statement": "Other claim.",
        "reason": "Wrong.",
        "verdict": 1}]}, ValueError, "cover every extracted statement", id="verdict-for-another-claim"),
        pytest.param({"statements": [{
        "statement": "Claim.",
        "reason": "Wrong.",
        "verdict": 2}]}, ValueError, "must be 0 or 1", id="verdict-out-of-range")])
@title("Scoring rejects judge verdicts that are missing, misattributed or out of range [{param_id}]")
def test_scoring_rejects_invalid_judge_verdicts(verdicts, error, message):
    judge, _ = scripted_judge({"statements": ["Claim."]}, verdicts)

    with pytest.raises(error, match=message):
        asyncio.run(score_sample(make_sample(), judge))


@title("A score that RAGAS reports inconsistently with the saved verdicts is rejected")
def test_scoring_rejects_score_inconsistent_with_verdicts(monkeypatch):
    collections = pytest.importorskip("ragas.metrics.collections")
    metric = Mock(ascore=AsyncMock(return_value=Mock(value=1.0)))
    monkeypatch.setattr(collections, "Faithfulness", Mock(return_value=metric))
    judge = Mock(
            calls=[{
            "output": {
            "statements": ["Claim."]}}, {
            "output": {
            "statements": [{
            "statement": "Claim.",
            "verdict": 0}]}}])

    with pytest.raises(ValueError, match="does not match its verdicts"):
        asyncio.run(score_sample(make_sample(), judge))


@title("Scoring requires exactly one extraction and one verification call")
def test_scoring_requires_two_judge_calls(monkeypatch):
    collections = pytest.importorskip("ragas.metrics.collections")
    metric = Mock(ascore=AsyncMock(return_value=Mock(value=1.0)))
    monkeypatch.setattr(collections, "Faithfulness", Mock(return_value=metric))
    judge = Mock(calls=[{"output": {"statements": ["Claim."]}}])

    with pytest.raises(ValueError, match="statement extraction and claim verification"):
        asyncio.run(score_sample(make_sample(), judge))


def rename_first_claim(name: str):
    """Rename the first claim consistently in statements and verdicts, so only the name itself is invalid."""
    def rename(result: dict) -> None:
        result["statements"][0] = name
        result["verdicts"][0]["statement"] = name

    return rename


@title("A result whose verdicts cover every statement and match the score is accepted unchanged")
def test_validate_result_accepts_consistent_result():
    result = make_result(labels=(1, 1, 0))

    assert validate_result(result) is result


@title("Score comparison tolerates floating-point rounding")
def test_validate_result_tolerates_rounding():
    result = make_result(labels=(1, 1, 0), value=2 / 3 + 1e-12)

    assert validate_result(result) is result


@pytest.mark.parametrize(
        ("corrupt", "message"),
        [
        pytest.param(
        lambda result: result.update(statements=tuple(result["statements"])), "cover every",
        id="statements-not-a-list"),
        # Without statements the score would be a division by zero.
        pytest.param(lambda result: result.update(statements=[], verdicts=[]), "cover every", id="no-statements"),
        pytest.param(rename_first_claim(" "), "cover every", id="blank-statement"),
        pytest.param(lambda result: result.update(verdicts=None), "cover every", id="verdicts-not-a-list"),
        pytest.param(
        lambda result: result["verdicts"].__setitem__(0, "Claim 0."), "cover every", id="verdict-not-a-dict"),
        pytest.param(lambda result: result["verdicts"][0].update(statement=0), "cover every", id="unnamed-verdict"),
        pytest.param(lambda result: result["verdicts"].pop(), "cover every", id="unverified-statement"),
        pytest.param(lambda result: result["verdicts"][0].update(verdict=True), "must be 0 or 1", id="boolean-verdict"),
        pytest.param(lambda result: result.update(value=0.6), "does not match its verdicts", id="score-differs")])
@title("Results with incomplete, misattributed or inconsistent verdicts are rejected [{param_id}]")
def test_validate_result_rejects_inconsistent_result(corrupt, message):
    result = make_result(labels=(1, 0))
    corrupt(result)

    with pytest.raises(ValueError, match=message):
        validate_result(result)


@title("Score comparison uses an absolute tolerance, so a tiny relative difference near zero is rejected")
def test_validate_result_compares_score_with_absolute_tolerance():
    result = make_result(labels=(0, 0), value=1e-6)

    with pytest.raises(ValueError, match="does not match its verdicts"):
        validate_result(result)


@title("A saved sample is reloaded from its observation and identified by the checksum of its bytes")
def test_load_sample_returns_sample_and_checksum(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(make_sample()))

    sample, checksum = load_sample(path)

    assert sample == make_sample()
    assert checksum == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(
        lambda sample: sample.update(retrieved_contexts=["An unrelated policy."]), "contexts do not match",
        id="substituted-context"),
        pytest.param(lambda sample: sample.update(schema_version=2), "contexts do not match", id="schema-version"),
        pytest.param(lambda sample: sample.update(response=" "), "nonempty response", id="blank-answer"),
        pytest.param(lambda sample: sample.update(reference=""), "nonempty reference", id="blank-reference")])
@title("A saved sample with edited contexts, an unknown schema or blank fields is rejected [{param_id}]")
def test_load_sample_rejects_edited_sample(tmp_path, corrupt, message):
    sample = make_sample()
    corrupt(sample)
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(sample))

    with pytest.raises(ValueError, match=message):
        load_sample(path)


@pytest.fixture
def offline_service(tmp_path, monkeypatch):
    """evaluate_sample_report with the Ollama transport, model catalog and judge replaced by doubles."""
    pytest.importorskip("ragas")
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(make_sample()))
    service = SimpleNamespace(sample_path=sample_path, transport=Mock(), ollama=Mock(), judge=Mock())
    service.ollama.list_models.return_value = model_catalog((TEST_MODEL, MODEL_DIGEST))
    service.judge.configure_mock(calls=[{"output": "raw judge call"}], options={"temperature": 0})
    service.http_class = Mock(return_value=service.transport)
    service.ollama_class = Mock(return_value=service.ollama)
    service.judge_class = Mock(return_value=service.judge)
    monkeypatch.setattr("llm_testkit.evaluation.faithfulness.HttpClient", service.http_class)
    monkeypatch.setattr("llm_testkit.evaluation.faithfulness.OllamaClient", service.ollama_class)
    monkeypatch.setattr("llm_testkit.evaluation.ollama_judge.OllamaJudge", service.judge_class)
    return service


def evaluate(service: SimpleNamespace) -> dict:
    return evaluate_sample_report(service.sample_path, settings=Settings(), judge_model=TEST_MODEL)


@title("A completed report records the measurement, its provenance and the judge, then closes the transport")
def test_report_records_completed_measurement(offline_service, monkeypatch):
    result = make_result()
    score = AsyncMock(return_value=result)
    monkeypatch.setattr(SCORE, score)
    settings = Settings()

    report = evaluate(offline_service)

    assert (report["schema_version"], report["metric"], report["status"]) == (1, "faithfulness", "completed")
    assert report["result"] == result
    assert report["threshold"] is None
    assert "not a calibrated quality gate" in report["interpretation"]
    assert report["ragas_version"] == version("ragas")
    assert datetime.fromisoformat(report["created_at"]).utcoffset() == timedelta(0)
    assert report["sample_path"] == str(offline_service.sample_path.resolve())
    assert report["sample_sha256"] == hashlib.sha256(offline_service.sample_path.read_bytes()).hexdigest()
    assert (report["generation_model"], report["same_generation_and_judge_model"]) == (TEST_MODEL, True)
    assert (report["judge_model"], report["judge_model_digest"]) == (TEST_MODEL, MODEL_DIGEST)
    assert report["judge_configuration"] == dict(options=offline_service.judge.options, think=False, retries=0)
    assert report["judge_calls"] == offline_service.judge.calls
    offline_service.http_class.assert_called_once_with(settings.ollama_base_url, settings.http_timeout)
    offline_service.ollama_class.assert_called_once_with(offline_service.transport)
    offline_service.judge_class.assert_called_once_with(offline_service.ollama, TEST_MODEL, settings.llm_timeout)
    score.assert_awaited_once_with(make_sample(), offline_service.judge)
    offline_service.transport.close.assert_called_once()


@title("A judge failure is reported with its error and raw calls, and the transport is still closed")
def test_report_preserves_judge_failure(offline_service, monkeypatch):
    monkeypatch.setattr(SCORE, AsyncMock(side_effect=ValueError("Judge generation truncated")))

    report = evaluate(offline_service)

    assert report["status"] == "error"
    assert report["error"] == {"type": "ValueError", "message": "Judge generation truncated"}
    assert report["judge_calls"] == offline_service.judge.calls
    offline_service.transport.close.assert_called_once()


@title("A judge model missing from the Ollama catalog is an error before the judge is created")
def test_report_requires_installed_judge_model(offline_service, monkeypatch):
    offline_service.ollama.list_models.return_value = model_catalog(("other-model", MODEL_DIGEST))
    monkeypatch.setattr(SCORE, AsyncMock(return_value=make_result()))

    report = evaluate(offline_service)

    assert report["status"] == "error"
    assert "Expected installed Ollama model" in report["error"]["message"]
    assert "judge_calls" not in report
    offline_service.judge_class.assert_not_called()
    offline_service.transport.close.assert_called_once()


@title("A failed Ollama catalog request is an error before the judge is created")
def test_report_requires_reachable_model_catalog(offline_service, monkeypatch):
    offline_service.ollama.list_models.return_value = model_catalog(status_code=503)
    monkeypatch.setattr(SCORE, AsyncMock(return_value=make_result()))

    report = evaluate(offline_service)

    assert "Judge model catalog: expected HTTP 200, got 503" in report["error"]["message"]
    offline_service.judge_class.assert_not_called()


@title("An unreadable sample is an error before any connection to Ollama; the default judge is still recorded")
def test_report_rejects_unreadable_sample_before_connecting(offline_service):
    offline_service.sample_path.write_text("{}")

    report = evaluate_sample_report(offline_service.sample_path, settings=Settings())

    assert (report["status"], report["error"]["type"]) == ("error", "KeyError")
    assert report["judge_model"] == "qwen3.5:4b"
    offline_service.http_class.assert_not_called()


@pytest.fixture
def cli_evaluation(monkeypatch):
    """The CLI's evaluation step, replaced so that argument handling and exit codes are checked offline."""
    monkeypatch.delenv("ANYTHINGLLM_API_KEY_FILE", raising=False)
    evaluation = Mock(return_value={"status": "completed", "result": {"value": 0.5}})
    monkeypatch.setattr("llm_testkit.evaluation.faithfulness.evaluate_sample_report", evaluation)
    return evaluation


@pytest.fixture
def run_cli(monkeypatch, cli_evaluation):
    def run(*arguments: str) -> int:
        monkeypatch.setattr("sys.argv", ["faithfulness", "sample.json", *arguments])
        return main()

    return run


@title("The CLI writes a completed report, prints the score and exits with 0")
def test_cli_writes_completed_report(run_cli, cli_evaluation, tmp_path, capsys):
    output = tmp_path / "report.json"

    exit_code = run_cli("--output", str(output))

    assert exit_code == 0
    assert json.loads(output.read_text()) == cli_evaluation.return_value
    assert "Faithfulness: 0.500" in capsys.readouterr().out
    cli_evaluation.assert_called_once_with(Path("sample.json"), settings=ANY, judge_model="qwen3.5:4b")
    assert isinstance(cli_evaluation.call_args.kwargs["settings"], Settings)


@title("The CLI passes the selected judge model to the evaluation")
def test_cli_passes_judge_model(run_cli, cli_evaluation, tmp_path):
    run_cli("--output", str(tmp_path / "report.json"), "--judge-model", "judge-model")

    assert cli_evaluation.call_args.kwargs["judge_model"] == "judge-model"


@title("The CLI saves a failed evaluation as an error report and exits with 1")
def test_cli_saves_failed_evaluation(run_cli, cli_evaluation, tmp_path, capsys):
    cli_evaluation.return_value = {"status": "error", "error": {"type": "KeyError"}}
    output = tmp_path / "report.json"

    exit_code = run_cli("--output", str(output))

    assert exit_code == 1
    assert json.loads(output.read_text()) == cli_evaluation.return_value
    assert "Evaluation failed" in capsys.readouterr().out


@title("The CLI refuses to overwrite an existing report before any model call")
def test_cli_refuses_existing_output(run_cli, cli_evaluation, tmp_path, capsys):
    output = tmp_path / "report.json"
    output.write_text("existing")

    with pytest.raises(SystemExit) as stopped:
        run_cli("--output", str(output))

    assert stopped.value.code == 2
    assert "Output already exists" in capsys.readouterr().err
    assert output.read_text() == "existing"
    cli_evaluation.assert_not_called()


@title("The CLI refuses to run without the evaluation dependencies")
def test_cli_requires_evaluation_dependencies(run_cli, cli_evaluation, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("llm_testkit.evaluation.faithfulness.import_module", Mock(side_effect=ImportError))

    with pytest.raises(SystemExit) as stopped:
        run_cli("--output", str(tmp_path / "report.json"))

    assert stopped.value.code == 2
    assert "Install evaluation dependencies" in capsys.readouterr().err
    cli_evaluation.assert_not_called()


@title("The CLI requires an output file")
def test_cli_requires_output(run_cli, cli_evaluation):
    with pytest.raises(SystemExit) as stopped:
        run_cli()

    assert stopped.value.code == 2
    cli_evaluation.assert_not_called()
