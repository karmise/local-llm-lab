"""Context precision and recall: RAGAS scoring, result validation, evidence integrity, the report service and the CLI."""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock, Mock, call

import pytest

from llm_testkit.config import Settings
from llm_testkit.evaluation.relevance import (
        check_relevance_evidence, evaluate_relevance_report, main, score_relevance, validate_relevance)
from llm_testkit.reporting.steps import title
from test_support.builders.golden import (
        GOLDEN_DATASET, GOLDEN_DATASET_FILE, PAID_LEAVE, POLICY_FILE, make_paid_leave_sample)
from test_support.builders.identities import MODEL_DIGEST, TEST_MODEL
from test_support.builders.ollama import chat_response, model_catalog
from test_support.builders.optional import load_ollama_judge
from test_support.builders.relevance import (
        ALLOWANCE_CLAIM, make_relevance_evidence, make_result, precision_verdicts, recall_classifications)

pytestmark = pytest.mark.unit

SCORE = "llm_testkit.evaluation.relevance.score_relevance"
THREE_CONTEXTS = ["Relevant", "Unrelated", "Relevant"]


def scripted_judge(outputs: list[dict], budget: int):
    """A real OllamaJudge whose transport replays the given outputs."""
    client = Mock()
    client.structured_chat.side_effect = [chat_response(output) for output in outputs]
    return load_ollama_judge()(client, TEST_MODEL, max_calls=budget), client


# Average precision rewards useful contexts that are retrieved early; recall is the share of attributed claims.


@pytest.mark.parametrize(("precision_labels", "expected_precision", "attributed", "expected_recall"), [
        pytest.param((1, 1, 0), 1.0, (1, 1), 1.0, id="useful-first"),
        pytest.param((1, 0, 1), 5 / 6, (1, 0), 0.5, id="interleaved"),
        pytest.param((0, 1, 1), 7 / 12, (0, 1), 0.5, id="useful-last"),
        pytest.param((0, 0, 0), 0.0, (0, 0), 0.0, id="nothing-useful")])
@title("Real RAGAS precision follows retrieval order and recall counts attributed reference claims [{param_id}]")
def test_ragas_relevance_scores(precision_labels, expected_precision, attributed, expected_recall):
    verdicts, classifications = precision_verdicts(precision_labels), recall_classifications(attributed)
    precision_judge, precision_client = scripted_judge(verdicts, budget=3)
    recall_judge, recall_client = scripted_judge([{"classifications": classifications}], budget=1)

    result = asyncio.run(
            score_relevance(make_paid_leave_sample(THREE_CONTEXTS), PAID_LEAVE, precision_judge, recall_judge))

    assert result["context_precision"] == pytest.approx(expected_precision)
    assert result["context_recall"] == expected_recall
    assert (result["precision_verdicts"], result["recall_classifications"]) == (verdicts, classifications)
    assert (precision_client.structured_chat.call_count, recall_client.structured_chat.call_count) == (3, 1)


@pytest.mark.parametrize("precision_labels", [(1, ), (1, 0, 1, 0)], ids=["one-context", "four-contexts"])
@title("Scoring accepts one to four contexts, with one precision call per context [{param_id}]")
def test_scoring_accepts_context_count_bounds(precision_labels):
    contexts = [f"Context {i}" for i in range(len(precision_labels))]
    precision_judge, precision_client = scripted_judge(precision_verdicts(precision_labels), budget=len(contexts))
    recall_judge, _ = scripted_judge([{"classifications": recall_classifications()}], budget=1)

    result = asyncio.run(score_relevance(make_paid_leave_sample(contexts), PAID_LEAVE, precision_judge, recall_judge))

    assert len(result["precision_verdicts"]) == len(contexts)
    assert precision_client.structured_chat.call_count == len(contexts)


@pytest.mark.parametrize("count", [0, 5])
@title("Scoring rejects zero or more than four contexts without judging or truncating them [{param_id}]")
def test_scoring_rejects_context_count_outside_bounds(count):
    pytest.importorskip("ragas")
    sample = {**make_paid_leave_sample(), "retrieved_contexts": ["Context"] * count}
    judge = Mock()

    with pytest.raises(ValueError, match="never truncated"):
        asyncio.run(score_relevance(sample, PAID_LEAVE, judge, judge))

    assert judge.mock_calls == []


@title("Scoring requires one precision call per context and exactly one recall call")
def test_scoring_requires_expected_judge_calls(monkeypatch):
    collections = pytest.importorskip("ragas.metrics.collections")
    metric = Mock(ascore=AsyncMock(return_value=Mock(value=1.0)))
    monkeypatch.setattr(collections, "ContextPrecisionWithReference", Mock(return_value=metric))
    monkeypatch.setattr(collections, "ContextRecall", Mock(return_value=metric))
    precision_judge, recall_judge = Mock(calls=[{}] * 2), Mock(calls=[{}])

    with pytest.raises(ValueError, match="Unexpected relevance judge call count"):
        asyncio.run(score_relevance(make_paid_leave_sample(THREE_CONTEXTS), PAID_LEAVE, precision_judge, recall_judge))


@pytest.mark.parametrize(("precision_labels", "expected_precision"), [((1, ), 1.0), ((1, 0, 1, 0), 5 / 6)],
        ids=["one-context", "four-contexts"])
@title("A result for one to four contexts with matching scores is accepted unchanged [{param_id}]")
def test_validate_accepts_one_to_four_contexts(precision_labels, expected_precision):
    result = make_result(precision_labels=precision_labels, context_precision=expected_precision)

    assert validate_relevance(result, PAID_LEAVE, len(precision_labels)) is result


@title("Labelled reference facts are found in recall claims regardless of letter case")
def test_validate_matches_labelled_facts_case_insensitively():
    result = make_result()
    for classification in result["recall_classifications"]:
        classification["statement"] = classification["statement"].upper()

    assert validate_relevance(result, PAID_LEAVE, 3) is result


@pytest.mark.parametrize(("corrupt", "context_count", "message"), [
        pytest.param(
        lambda result: result["precision_verdicts"].pop(), 3, "one verdict per captured context",
        id="missing-context-verdict"),
        pytest.param(
        lambda result: result.update(precision_verdicts=[]), 0, "one verdict per captured context", id="no-contexts"),
        pytest.param(
        lambda result: result.update(precision_verdicts=precision_verdicts(
        (1, 0, 1, 0, 1))), 5, "one verdict per captured context, maximum four", id="five-contexts"),
        pytest.param(
        lambda result: result.update(recall_classifications=[]), 3, "nonempty and unique", id="no-reference-claims"),
        pytest.param(
        lambda result: result.update(
        recall_classifications=recall_classifications((1, 1), statements=(ALLOWANCE_CLAIM, ALLOWANCE_CLAIM))), 3,
        "nonempty and unique", id="duplicate-reference-claim"),
        pytest.param(
        lambda result: result["precision_verdicts"][0].update(verdict=True), 3, "binary integers",
        id="boolean-precision-verdict"),
        pytest.param(
        lambda result: result["recall_classifications"][0].update(attributed=2), 3, "binary integers",
        id="attribution-out-of-range"),
        pytest.param(
        lambda result: result["precision_verdicts"][0].pop("reason"), 3, "nonempty reasons",
        id="unexplained-precision-verdict"),
        pytest.param(
        lambda result: result["recall_classifications"][0].update(reason=" "), 3, "nonempty reasons",
        id="blank-recall-reason"),
        pytest.param(
        lambda result: result["recall_classifications"][1].update(statement="Some deadline."), 3,
        "omit a labelled reference fact: advance notice", id="missing-notice-fact"),
        pytest.param(
        lambda result: result.update(context_precision=1.0), 3, "context_precision does not match",
        id="precision-differs"),
        pytest.param(
        lambda result: result.update(context_recall=1.0), 3, "context_recall does not match", id="recall-differs")])
@title("Results with missing, non-binary, unexplained or inconsistent verdicts are rejected [{param_id}]")
def test_validate_rejects_inconsistent_result(corrupt, context_count, message):
    result = make_result()
    corrupt(result)

    with pytest.raises(ValueError, match=message):
        validate_relevance(result, PAID_LEAVE, context_count)


@title("A score that is not a finite number from 0 to 1 is rejected")
def test_validate_rejects_invalid_score():
    with pytest.raises(AssertionError, match="finite quality score"):
        validate_relevance(make_result(context_recall=float("nan")), PAID_LEAVE, 3)


@title("Saved relevance evidence is accepted when bound to its sample, case and raw judge calls")
def test_evidence_check_returns_validated_result():
    evidence = make_relevance_evidence()

    checked = check_relevance_evidence(evidence, "sample", make_paid_leave_sample(THREE_CONTEXTS), GOLDEN_DATASET)

    assert checked == evidence["result"]


@pytest.mark.parametrize(("changes", "message"), [
        pytest.param({"schema_version": 2}, "completed context evidence", id="schema-version"),
        pytest.param({"metric": "faithfulness"}, "completed context evidence", id="other-metric"),
        pytest.param({"status": "error"}, "completed context evidence", id="unfinished"),
        pytest.param({"sample_sha256": "other"}, "checksum mismatch", id="other-sample"),
        pytest.param({"golden_dataset_sha256": "stale"}, "checksum mismatch", id="stale-dataset"),
        pytest.param({"golden_case_id": "not_in_dataset"}, "Unknown golden case", id="unknown-case"),
        pytest.param({"question": "Edited"}, "reference/question mismatch", id="question"),
        pytest.param({"reference": "Edited"}, "reference/question mismatch", id="reference"),
        pytest.param({"result": make_result(context_recall=1.0)}, "context_recall does not match", id="inconsistent")])
@title("Relevance evidence that is unfinished or bound to other inputs is rejected [{param_id}]")
def test_evidence_check_rejects_unbound_evidence(changes, message):
    evidence = {**make_relevance_evidence(), **changes}

    with pytest.raises(ValueError, match=message):
        check_relevance_evidence(evidence, "sample", make_paid_leave_sample(THREE_CONTEXTS), GOLDEN_DATASET)


@pytest.mark.parametrize(
        "corrupt", [
        pytest.param(
        lambda evidence: evidence["precision_calls"][0]["output"].update(verdict=0), id="precision-verdict"),
        pytest.param(lambda evidence: evidence["precision_calls"].pop(), id="missing-precision-call"),
        pytest.param(lambda evidence: evidence["recall_calls"].append({}), id="extra-recall-call"),
        pytest.param(
        lambda evidence: evidence["recall_calls"][0]["output"]["classifications"][0].update(reason="Edited"),
        id="recall-classification")])
@title("A summary that differs from the saved raw judge calls is rejected [{param_id}]")
def test_evidence_check_rejects_summary_that_differs_from_raw_calls(corrupt):
    evidence = make_relevance_evidence()
    corrupt(evidence)

    with pytest.raises(ValueError, match="differs from raw judge calls"):
        check_relevance_evidence(evidence, "sample", make_paid_leave_sample(THREE_CONTEXTS), GOLDEN_DATASET)


@pytest.fixture
def offline_service(tmp_path, monkeypatch):
    """evaluate_relevance_report with the Ollama transport, model catalog and both judges replaced by doubles."""
    pytest.importorskip("ragas")
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(make_paid_leave_sample(THREE_CONTEXTS)))
    service = SimpleNamespace(sample_path=sample_path, transport=Mock(), ollama=Mock())
    service.ollama.list_models.return_value = model_catalog((TEST_MODEL, MODEL_DIGEST))
    service.precision_judge = Mock(calls=[{"output": "precision call"}], options={"temperature": 0})
    service.recall_judge = Mock(calls=[{"output": "recall call"}], options={"temperature": 0})
    service.http_class = Mock(return_value=service.transport)
    service.ollama_class = Mock(return_value=service.ollama)
    service.judge_class = Mock(side_effect=[service.precision_judge, service.recall_judge])
    monkeypatch.setattr("llm_testkit.evaluation.relevance.HttpClient", service.http_class)
    monkeypatch.setattr("llm_testkit.evaluation.relevance.OllamaClient", service.ollama_class)
    monkeypatch.setattr("llm_testkit.evaluation.ollama_judge.OllamaJudge", service.judge_class)
    return service


def evaluate(service: SimpleNamespace, *, case_id: str = PAID_LEAVE.id) -> dict:
    return evaluate_relevance_report(
            service.sample_path, dataset_path=GOLDEN_DATASET_FILE, policy_file=POLICY_FILE, case_id=case_id,
            settings=Settings(), judge_model=TEST_MODEL)


@title("A completed report records both measurements, their provenance and both judges, then closes the transport")
def test_report_records_completed_measurement(offline_service, monkeypatch):
    result = make_result()
    score = AsyncMock(return_value=result)
    monkeypatch.setattr(SCORE, score)
    settings = Settings()
    expected_configuration = dict(options={"temperature": 0}, think=False, retries=0, maximum_calls=4)

    report = evaluate(offline_service)

    assert (report["schema_version"], report["metric"], report["status"]) == (1, "context_relevance", "completed")
    assert report["result"] == result
    assert report["threshold"] is None
    assert "exploratory measurements" in report["interpretation"]
    assert report["ragas_version"] == version("ragas")
    assert datetime.fromisoformat(report["created_at"]).utcoffset() == timedelta(0)
    assert report["sample_sha256"] == hashlib.sha256(offline_service.sample_path.read_bytes()).hexdigest()
    assert (report["golden_case_id"], report["golden_dataset_sha256"]) == (PAID_LEAVE.id, GOLDEN_DATASET.sha256)
    assert (report["question"], report["reference"]) == (PAID_LEAVE.question, PAID_LEAVE.reference)
    assert (report["judge_model"], report["judge_model_digest"]) == (TEST_MODEL, MODEL_DIGEST)
    assert report["judge_configuration"] == expected_configuration
    assert report["precision_calls"] == offline_service.precision_judge.calls
    assert report["recall_calls"] == offline_service.recall_judge.calls
    offline_service.http_class.assert_called_once_with(settings.ollama_base_url, settings.http_timeout)
    offline_service.ollama_class.assert_called_once_with(offline_service.transport)
    assert offline_service.judge_class.call_args_list == [
            call(offline_service.ollama, TEST_MODEL, settings.llm_timeout, max_calls=3),
            call(offline_service.ollama, TEST_MODEL, settings.llm_timeout, max_calls=1)]
    score.assert_awaited_once_with(
            make_paid_leave_sample(THREE_CONTEXTS), PAID_LEAVE, offline_service.precision_judge,
            offline_service.recall_judge)
    offline_service.transport.close.assert_called_once()


@title("A judge failure is reported with its error and both judges' raw calls, and the transport is still closed")
def test_report_preserves_judge_failure(offline_service, monkeypatch):
    monkeypatch.setattr(SCORE, AsyncMock(side_effect=ValueError("Truncated response")))

    report = evaluate(offline_service)

    assert report["status"] == "error"
    assert report["error"] == {"type": "ValueError", "message": "Truncated response"}
    assert report["precision_calls"] == offline_service.precision_judge.calls
    assert report["recall_calls"] == offline_service.recall_judge.calls
    offline_service.transport.close.assert_called_once()


@pytest.mark.parametrize("count", [1, 4])
@title("The report accepts one to four contexts and budgets one precision call per context [{param_id}]")
def test_report_accepts_context_count_bounds(offline_service, monkeypatch, count):
    offline_service.sample_path.write_text(json.dumps(make_paid_leave_sample([f"Context {i}" for i in range(count)])))
    monkeypatch.setattr(SCORE, AsyncMock(return_value=make_result()))

    report = evaluate(offline_service)

    assert report["status"] == "completed"
    assert report["judge_configuration"]["maximum_calls"] == count + 1
    assert offline_service.judge_class.call_args_list[0].kwargs == {"max_calls": count}


@title("A judge model missing from the Ollama catalog is an error before the judges are created")
def test_report_requires_installed_judge_model(offline_service, monkeypatch):
    offline_service.ollama.list_models.return_value = model_catalog(("other-model", MODEL_DIGEST))
    monkeypatch.setattr(SCORE, AsyncMock(return_value=make_result()))

    report = evaluate(offline_service)

    assert report["status"] == "error"
    assert "Expected installed Ollama model" in report["error"]["message"]
    assert (report["precision_calls"], report["recall_calls"]) == ([], [])
    offline_service.judge_class.assert_not_called()
    offline_service.transport.close.assert_called_once()


@title("A failed Ollama catalog request is an error before the judges are created")
def test_report_requires_reachable_model_catalog(offline_service, monkeypatch):
    offline_service.ollama.list_models.return_value = model_catalog(status_code=503)
    monkeypatch.setattr(SCORE, AsyncMock(return_value=make_result()))

    report = evaluate(offline_service)

    assert "Relevance judge catalog: expected HTTP 200, got 503" in report["error"]["message"]
    offline_service.judge_class.assert_not_called()


@title("More than four captured contexts are an error before any connection; the default judge is still recorded")
def test_report_rejects_too_many_contexts_before_connecting(offline_service):
    offline_service.sample_path.write_text(json.dumps(make_paid_leave_sample(["Context"] * 5)))

    report = evaluate_relevance_report(
            offline_service.sample_path, dataset_path=GOLDEN_DATASET_FILE, policy_file=POLICY_FILE,
            case_id=PAID_LEAVE.id, settings=Settings())

    assert (report["status"], report["judge_model"]) == ("error", "qwen3.5:4b")
    assert "never truncated" in report["error"]["message"]
    offline_service.http_class.assert_not_called()


@title("A sample that does not belong to the selected golden case is an error before any connection")
def test_report_rejects_sample_for_other_case(offline_service):
    report = evaluate(offline_service, case_id="carryover_limit")

    assert report["status"] == "error"
    assert "question/reference does not match" in report["error"]["message"]
    offline_service.http_class.assert_not_called()


@pytest.fixture
def cli_evaluation(monkeypatch):
    """The CLI's evaluation step, replaced so that argument handling and exit codes are checked offline."""
    monkeypatch.delenv("ANYTHINGLLM_API_KEY_FILE", raising=False)
    evaluation = Mock(return_value={"status": "completed"})
    monkeypatch.setattr("llm_testkit.evaluation.relevance.evaluate_relevance_report", evaluation)
    return evaluation


@pytest.fixture
def run_cli(monkeypatch, cli_evaluation):
    def run(*arguments: str) -> int:
        monkeypatch.setattr("sys.argv", ["relevance", "sample.json", "--case", PAID_LEAVE.id, *arguments])
        return main()

    return run


@title("The CLI evaluates the selected case with default inputs and writes the report to a new file")
def test_cli_writes_completed_report(run_cli, cli_evaluation, tmp_path, capsys):
    output = tmp_path / "relevance.json"

    exit_code = run_cli("--output", str(output))

    assert exit_code == 0
    assert json.loads(output.read_text()) == {"status": "completed"}
    assert "Relevance: completed" in capsys.readouterr().out
    cli_evaluation.assert_called_once_with(
            Path("sample.json"), dataset_path=Path("test_data/golden-policy.json"),
            policy_file=Path("test_data/company-policy.txt"), case_id=PAID_LEAVE.id, settings=ANY,
            judge_model="qwen3.5:4b")
    assert isinstance(cli_evaluation.call_args.kwargs["settings"], Settings)


@title("The CLI passes the chosen dataset, policy and judge model to the evaluation")
def test_cli_passes_selected_inputs(run_cli, cli_evaluation, tmp_path):
    run_cli(
            "--output", str(tmp_path / "out.json"), "--dataset", "dataset.json", "--policy", "policy.txt",
            "--judge-model", "judge-model")

    arguments = cli_evaluation.call_args.kwargs
    assert (arguments["dataset_path"], arguments["policy_file"]) == (Path("dataset.json"), Path("policy.txt"))
    assert arguments["judge_model"] == "judge-model"


@title("The CLI exits with 1 when the evaluation fails")
def test_cli_exit_code_reports_failure(run_cli, cli_evaluation, tmp_path):
    cli_evaluation.return_value = {"status": "error"}

    assert run_cli("--output", str(tmp_path / "out.json")) == 1


@title("The CLI refuses to overwrite existing evidence before any model call")
def test_cli_refuses_existing_output(run_cli, cli_evaluation, tmp_path, capsys):
    output = tmp_path / "existing.json"
    output.write_text("existing")

    with pytest.raises(SystemExit) as stopped:
        run_cli("--output", str(output))

    assert stopped.value.code == 2
    assert "Output already exists" in capsys.readouterr().err
    assert output.read_text() == "existing"
    cli_evaluation.assert_not_called()


@pytest.mark.parametrize(
        "arguments", [
        pytest.param(["sample.json", "--output", "out.json"], id="without-case"),
        pytest.param(["sample.json", "--case", "paid_leave"], id="without-output")])
@title("The CLI requires the golden case and an output file [{param_id}]")
def test_cli_requires_case_and_output(cli_evaluation, monkeypatch, tmp_path, arguments):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["relevance", *arguments])

    with pytest.raises(SystemExit) as stopped:
        main()

    assert stopped.value.code == 2
    cli_evaluation.assert_not_called()
