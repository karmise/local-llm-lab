"""Faithfulness: sentence-level claim checks, saved-sample integrity, the report service and the CLI."""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock

import pytest
import requests

from builders.faithfulness import CONTEXT, make_result, make_sample
from builders.identities import MODEL_DIGEST, TEST_MODEL
from builders.ollama import model_catalog, text_reply
from config import Settings
from evaluation.claim_checker import CHECKER_MODEL, MAX_CLAIMS, ClaimChecker
from evaluation.faithfulness import evaluate_sample_report, load_sample, main, score_sample, validate_result
from reporting.steps import title

pytestmark = pytest.mark.unit

SCORE = "evaluation.faithfulness.score_sample"
ANSWER = "Supported claim. Unsupported claim."


def scripted_checker(*replies, max_calls: int = MAX_CLAIMS):
    """A real ClaimChecker whose transport replays the given chat replies."""
    client = Mock()
    client.chat.side_effect = list(replies)
    return ClaimChecker(client, TEST_MODEL, 30, max_calls=max_calls), client


def sample_answering(answer: str) -> dict:
    return {**make_sample(), "response": answer}


@title("Faithfulness is the share of answer sentences the checker finds supported, each checked once as written")
def test_faithfulness_is_share_of_supported_sentences():
    checker, client = scripted_checker(text_reply("Yes"), text_reply("No"))

    result = asyncio.run(score_sample(sample_answering(ANSWER), checker))

    assert result == {
            "value": 0.5,
            "statements": ["Supported claim.", "Unsupported claim."],
            "verdicts": [{
            "statement": "Supported claim.",
            "verdict": 1}, {
            "statement": "Unsupported claim.",
            "verdict": 0}]}
    prompts = [call.kwargs["prompt"] for call in client.chat.call_args_list]
    assert prompts == [
            f"Document: {CONTEXT}\nClaim: Supported claim.", f"Document: {CONTEXT}\nClaim: Unsupported claim."]
    assert {call.kwargs["options"]["temperature"] for call in client.chat.call_args_list} == {0}
    assert [(call["claim"], call["verdict"]) for call in checker.calls] == [("Supported claim.", 1),
            ("Unsupported claim.", 0)]


@title("An answer with more sentences than the claim budget is rejected before any checker call")
def test_scoring_rejects_answer_over_budget():
    checker, client = scripted_checker(max_calls=2)

    with pytest.raises(ValueError, match="3 sentences; the claim budget is 2"):
        asyncio.run(score_sample(sample_answering("One. Two. Three."), checker))
    client.chat.assert_not_called()


@title("An answer without a sentence to check is rejected before any checker call")
def test_scoring_rejects_answer_without_sentences():
    checker, client = scripted_checker()

    with pytest.raises(ValueError, match="no sentence to check"):
        asyncio.run(score_sample(sample_answering("..."), checker))
    client.chat.assert_not_called()


@pytest.mark.parametrize(("reply", "message"), [
        pytest.param(text_reply("Maybe"), "must answer Yes or No, got 'Maybe'", id="not-yes-or-no"),
        pytest.param(text_reply("yes"), "must answer Yes or No", id="lowercase"),
        pytest.param(text_reply("Yes", done_reason="length"), "incomplete or truncated", id="truncated"),
        pytest.param(text_reply("Yes", done=False), "incomplete or truncated", id="unfinished"),
        pytest.param(text_reply("Yes", model="other-model"), "unexpected model", id="other-model")])
@title("A checker reply that is not a complete Yes or No from the selected model is an error [{param_id}]")
def test_checker_rejects_invalid_reply(reply, message):
    checker, _ = scripted_checker(reply)

    with pytest.raises(ValueError, match=message):
        checker.check("Document", "Claim.")
    assert checker.calls[0]["ollama_response"]["message"]["content"] in ("Maybe", "yes", "Yes")


@title("A failed checker request raises its HTTP error and keeps the attempted call")
def test_checker_raises_http_error():
    checker, _ = scripted_checker(text_reply("Yes", status_code=503))

    with pytest.raises(requests.HTTPError):
        checker.check("Document", "Claim.")
    assert checker.calls == [{"claim": "Claim.", "prompt": "Document: Document\nClaim: Claim."}]


@title("The checker refuses a call beyond its budget before contacting Ollama")
def test_checker_budget_is_enforced_before_transport():
    checker, client = scripted_checker(text_reply("Yes"), max_calls=1)
    checker.check("Document", "First.")

    with pytest.raises(ValueError, match="budget exceeded"):
        checker.check("Document", "Second.")
    assert client.chat.call_count == 1


@pytest.mark.parametrize("budget", [0, MAX_CLAIMS + 1, True, 1.0])
@title("A claim budget outside one to six calls is rejected [{budget}]")
def test_checker_rejects_invalid_budget(budget):
    with pytest.raises(ValueError, match="between one and"):
        ClaimChecker(Mock(), TEST_MODEL, max_calls=budget)


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
    """evaluate_sample_report with the Ollama transport, model catalog and claim checker replaced by doubles."""
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(make_sample()))
    service = SimpleNamespace(sample_path=sample_path, transport=Mock(), ollama=Mock(), judge=Mock())
    service.ollama.list_models.return_value = model_catalog((TEST_MODEL, MODEL_DIGEST))
    service.judge.configure_mock(calls=[{"output": "raw judge call"}], options={"temperature": 0})
    service.http_class = Mock(return_value=service.transport)
    service.ollama_class = Mock(return_value=service.ollama)
    service.judge_class = Mock(return_value=service.judge)
    monkeypatch.setattr("evaluation.judge_connection.HttpClient", service.http_class)
    monkeypatch.setattr("evaluation.judge_connection.OllamaClient", service.ollama_class)
    monkeypatch.setattr("evaluation.faithfulness.ClaimChecker", service.judge_class)
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

    assert (report["schema_version"], report["metric"], report["status"]) == (2, "faithfulness", "completed")
    assert report["result"] == result
    assert report["threshold"] is None
    assert "not a calibrated quality gate" in report["interpretation"]
    assert report["method"] == "sentence split in code; one grounding check per sentence"
    assert datetime.fromisoformat(report["created_at"]).utcoffset() == timedelta(0)
    assert report["sample_path"] == str(offline_service.sample_path.resolve())
    assert report["sample_sha256"] == hashlib.sha256(offline_service.sample_path.read_bytes()).hexdigest()
    assert (report["generation_model"], report["same_generation_and_judge_model"]) == (TEST_MODEL, True)
    assert (report["judge_model"], report["judge_model_digest"]) == (TEST_MODEL, MODEL_DIGEST)
    assert report["judge_configuration"] == dict(options=offline_service.judge.options, retries=0)
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
    assert report["judge_model"] == CHECKER_MODEL
    offline_service.http_class.assert_not_called()


@pytest.fixture
def cli_evaluation(monkeypatch):
    """The CLI's evaluation step, replaced so that argument handling and exit codes are checked offline."""
    monkeypatch.delenv("ANYTHINGLLM_API_KEY_FILE", raising=False)
    evaluation = Mock(return_value={"status": "completed", "result": {"value": 0.5}})
    monkeypatch.setattr("evaluation.faithfulness.evaluate_sample_report", evaluation)
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
    cli_evaluation.assert_called_once_with(Path("sample.json"), settings=ANY, judge_model=CHECKER_MODEL)
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


@title("The CLI requires an output file")
def test_cli_requires_output(run_cli, cli_evaluation):
    with pytest.raises(SystemExit) as stopped:
        run_cli()

    assert stopped.value.code == 2
    cli_evaluation.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), 1.5, -0.1, True, None])
@title("A saved faithfulness score that is not a finite number from 0 to 1 is a validation error [{value}]")
def test_validate_result_rejects_invalid_score(value):
    result = {"value": value, "statements": ["Claim."], "verdicts": [{"statement": "Claim.", "verdict": 1}]}

    with pytest.raises(ValueError, match="Faithfulness score must be a finite number between 0 and 1"):
        validate_result(result)
