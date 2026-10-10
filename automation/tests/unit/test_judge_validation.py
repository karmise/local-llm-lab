"""Judge validation: real RAGAS on labelled controls, raw-evidence checks, label comparison, the batch and the CLI."""

import asyncio
import json
from datetime import datetime, timedelta
from importlib.metadata import version
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import pytest

from llm_testkit.config import Settings
from llm_testkit.datasets.judge_controls import maximum_calls
from llm_testkit.evaluation import judge_validation
from llm_testkit.evaluation.judge_validation import (
        evaluate_judge_controls, label_mismatches, main, result_from_evidence, score_control, summarize_validation)
from llm_testkit.reporting.steps import title
from test_support.builders.golden import GOLDEN_DATASET
from test_support.builders.judge_validation import (
        JUDGE_CONTROLS_FILE, RECALL_STATEMENTS, control, control_outputs, curated_controls)
from test_support.builders.ollama import chat_response, model_catalog
from test_support.builders.optional import load_ollama_judge
from test_support.data.common import MODEL_DIGEST, TEST_MODEL
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit

CURATED_IDS = [case["id"] for case in curated_controls().cases]
PRECISION_FIRST = 1 / (1 + 1e-10)


def raw_calls(case: dict) -> list[dict]:
    return [{"output": output} for output in control_outputs(case)]


@pytest.mark.parametrize("control_id", CURATED_IDS)
@title("Real RAGAS scoring of a control's labelled judge replies reproduces its labels and score [{param_id}]")
def test_real_metric_reproduces_control_labels(control_id):
    case = control(control_id)
    client = Mock()
    client.structured_chat.side_effect = [chat_response(output) for output in control_outputs(case)]
    judge = load_ollama_judge()(client, TEST_MODEL, max_calls=maximum_calls(case))

    (row, ) = asyncio.run(evaluate_judge_controls([case], lambda budget: judge))

    assert (row["id"], row["metric"], row["status"], row["mismatches"]) == (control_id, case["metric"], "matched", [])
    assert row["result"]["value"] == pytest.approx(case["expected_score"], abs=1e-8)
    assert row["judge_calls"] == judge.calls
    assert client.structured_chat.call_count == maximum_calls(case)


@pytest.mark.parametrize(("control_id", "value", "expected"), [
        pytest.param("receipt_condition_supported", 1.0, {"response": [1]}, id="faithfulness"),
        pytest.param("manager_wrong_role", 0.0, {
        "response": [0],
        "reference": [0]}, id="correctness"),
        pytest.param("relevant_context_second", 0.5 / (1 + 1e-10), {"verdicts": [0, 1]}, id="precision"),
        pytest.param("notice_fact_not_retrieved", 0.5, {"reference": [1, 0]}, id="recall")])
@title("Raw judge calls are reduced to the score and the claim verdicts each metric is labelled with [{param_id}]")
def test_result_from_evidence(control_id, value, expected):
    case = control(control_id)

    result = result_from_evidence(case, raw_calls(case), value)

    assert result["value"] == value
    assert set(result) == {"value", *expected}
    for field, labels in expected.items():
        verdicts = result[field] if field == "verdicts" else [row["verdict"] for row in result[field]]
        assert verdicts == labels


@title("Recall evidence keeps each classification with its attribution as the verdict")
def test_result_from_evidence_keeps_recall_rows():
    case = control("notice_fact_not_retrieved")

    result = result_from_evidence(case, raw_calls(case), 0.5)

    assert [row["statement"] for row in result["reference"]] == list(RECALL_STATEMENTS)
    assert all(row["verdict"] == row["attributed"] for row in result["reference"])


def corrupt_output(index: int, **changes):
    return lambda calls: calls[index]["output"].update(changes)


def corrupt_recall(**changes):
    return lambda calls: calls[0]["output"]["classifications"][0].update(changes)


@pytest.mark.parametrize(("control_id", "value", "corrupt", "error", "message"), [
        pytest.param(
        "relevant_context_first", PRECISION_FIRST, lambda calls: calls.pop(), ValueError, "unexpected number of calls",
        id="missing-call"),
        pytest.param(
        "relevant_context_first", float("nan"), lambda calls: None, AssertionError, "finite quality score",
        id="invalid-score"),
        pytest.param(
        "relevant_context_first", 0.5, lambda calls: None, ValueError, "disagrees with raw verdicts",
        id="forged-precision"),
        pytest.param(
        "relevant_context_first", PRECISION_FIRST, corrupt_output(1, verdict=2), ValueError,
        "binary and include reasons", id="precision-verdict-two"),
        pytest.param(
        "relevant_context_first", PRECISION_FIRST, corrupt_output(0, verdict=True), ValueError,
        "binary and include reasons", id="boolean-precision-verdict"),
        pytest.param(
        "relevant_context_first", PRECISION_FIRST, corrupt_output(0, reason=" "), ValueError,
        "binary and include reasons", id="unexplained-precision-verdict"),
        pytest.param(
        "notice_fact_not_retrieved", 1.0, lambda calls: None, ValueError, "disagrees with raw verdicts",
        id="forged-recall"),
        pytest.param(
        "notice_fact_not_retrieved", 0.5, corrupt_recall(statement=RECALL_STATEMENTS[1]), ValueError,
        "nonempty, unique claim evidence", id="duplicate-recall-claim"),
        pytest.param(
        "notice_fact_not_retrieved", 0.5, lambda calls: calls[0]["output"].update(classifications=[]), ValueError,
        "nonempty, unique claim evidence", id="no-recall-claims"),
        pytest.param(
        "notice_fact_not_retrieved", 0.5, corrupt_recall(reason=""), ValueError, "binary and include reasons",
        id="unexplained-recall-claim"),
        pytest.param(
        "receipt_condition_supported", 0.0, lambda calls: None, ValueError, "does not match its verdicts",
        id="forged-faithfulness"),
        pytest.param(
        "manager_policy_paraphrase", 0.5, lambda calls: None, ValueError, "F1 does not match", id="forged-correctness")
])
@title("Raw evidence with missing calls, invalid verdicts or a forged score is rejected [{param_id}]")
def test_result_from_evidence_rejects_invalid_evidence(control_id, value, corrupt, error, message):
    case = control(control_id)
    calls = raw_calls(case)
    corrupt(calls)

    with pytest.raises(error, match=message):
        result_from_evidence(case, calls, value)


def recall_result(*rows: tuple[str, int], value: float = 0.5) -> dict:
    return {"value": value, "reference": [{"statement": s, "verdict": v} for s, v in rows]}


@title("A result that reproduces the labels has no mismatches")
def test_labels_match():
    rows = [(RECALL_STATEMENTS[0], 1), (RECALL_STATEMENTS[1], 0)]

    assert label_mismatches(recall_result(*rows), control("notice_fact_not_retrieved")) == []


@title("Claim labels are matched regardless of letter case")
def test_labels_match_case_insensitively():
    rows = [(RECALL_STATEMENTS[0].upper(), 1), (RECALL_STATEMENTS[1].upper(), 0)]

    assert label_mismatches(recall_result(*rows), control("notice_fact_not_retrieved")) == []


@pytest.mark.parametrize(("value", "mismatched"), [(0.5 + 5e-9, False), (0.5 + 2e-8, True)],
        ids=["within-tolerance", "outside-tolerance"])
@title("Scores are compared with an absolute tolerance of 1e-8 [{param_id}]")
def test_label_score_tolerance(value, mismatched):
    rows = [(RECALL_STATEMENTS[0], 1), (RECALL_STATEMENTS[1], 0)]

    mismatches = label_mismatches(recall_result(*rows, value=value), control("notice_fact_not_retrieved"))

    assert bool(mismatches) is mismatched


@title("A matching score cannot hide reversed claim verdicts")
def test_reversed_verdicts_with_same_score_mismatch():
    case = control("notice_fact_not_retrieved")
    rows = [(RECALL_STATEMENTS[0], 0), (RECALL_STATEMENTS[1], 1)]

    mismatches = label_mismatches(recall_result(*rows), case)

    assert mismatches == [f"reference claim label disagrees: {rule['pattern']}" for rule in case["labels"]["reference"]]


@title("A different score is reported with the expected and observed values")
def test_score_mismatch_message():
    rows = [(RECALL_STATEMENTS[0], 1), (RECALL_STATEMENTS[1], 1)]

    assert label_mismatches(recall_result(*rows, value=1.0),
            control("notice_fact_not_retrieved"))[0] == ("Expected score 0.5, observed 1.0")


@title("A labelled fact without a matching claim, and a claim without a label, both need review")
def test_missing_and_unlabelled_claims_mismatch():
    case = control("notice_fact_not_retrieved")
    rows = [(RECALL_STATEMENTS[0], 1), ("The manager approves the request.", 0)]

    mismatches = label_mismatches(recall_result(*rows), case)

    assert mismatches == [
            f"reference claim label disagrees: {case['labels']['reference'][1]['pattern']}",
            "Unlabelled reference claims require review"]


@title("Precision verdicts must match the labels in retrieval order")
def test_precision_order_mismatch():
    mismatches = label_mismatches({"value": 1.0, "verdicts": [0, 1]}, control("relevant_context_first"))

    assert "Ordered context verdicts differ from labels" in mismatches


@title("Precision verdicts in the labelled order have no mismatches beyond the score")
def test_precision_order_match():
    assert label_mismatches({"value": 1.0, "verdicts": [1, 0]}, control("relevant_context_first")) == []


@pytest.mark.parametrize(
        "cases",
        [pytest.param([], id="empty"),
        pytest.param([control("manager_policy_paraphrase")] * 9, id="thirty-six-calls")])
@title("An empty batch or one exceeding 32 judge calls is rejected before any judge is created [{param_id}]")
def test_batch_rejects_unbounded_batch(cases):
    factory = Mock()

    with pytest.raises(ValueError, match="bounded nonempty batch"):
        asyncio.run(evaluate_judge_controls(cases, factory))

    factory.assert_not_called()


@title("A batch of exactly 32 judge calls is accepted")
def test_batch_accepts_largest_batch():
    case = control("manager_policy_paraphrase")
    scorer = AsyncMock(side_effect=ValueError("Not scored in this test"))

    results = asyncio.run(evaluate_judge_controls([case] * 8, Mock(return_value=Mock(calls=[])), scorer))

    assert len(results) == 8


@title("Each control gets a fresh judge with its own budget; mismatches and errors do not stop the batch")
def test_batch_keeps_independent_outcomes():
    cases = [
            control(i) for i in (
            "receipt_condition_supported", "receipt_condition_contradicted", "manager_policy_paraphrase",
            "manager_wrong_role")]
    judges = [Mock(calls=raw_calls(case)) for case in cases]
    # The judge supports the contradicted claim: consistent evidence that disagrees with the labels.
    judges[1].calls[1]["output"]["statements"][0]["verdict"] = 1
    judges[2].calls = [{"error": "truncated"}]
    factory = Mock(side_effect=judges)
    scores = [1.0, 1.0, ValueError("Truncated judge evidence"), 0.0]
    scorer = AsyncMock(side_effect=scores)

    results = asyncio.run(evaluate_judge_controls(cases, factory, scorer))

    assert [row["status"] for row in results] == ["matched", "mismatch", "error", "matched"]
    assert [row["control"] for row in results] == cases
    assert results[1]["mismatches"][0] == "Expected score 0, observed 1.0"
    assert results[2]["error"] == {"type": "ValueError", "message": "Truncated judge evidence"}
    assert [row["judge_calls"] for row in results] == [judge.calls for judge in judges]
    assert factory.call_args_list == [call(2), call(2), call(4), call(4)]
    assert [awaited.args for awaited in scorer.await_args_list] == list(zip(cases, judges, strict=True))


@title("A judge that cannot be created is an error with no recorded calls")
def test_batch_records_failed_judge_creation():
    (row, ) = asyncio.run(
            evaluate_judge_controls([control("receipt_condition_supported")],
            Mock(side_effect=RuntimeError("Ollama offline")), AsyncMock()))

    assert (row["status"], row["judge_calls"]) == ("error", [])
    assert row["error"] == {"type": "RuntimeError", "message": "Ollama offline"}


@pytest.mark.parametrize(("control_id", "target"), [("receipt_condition_supported", "score_sample"),
        ("manager_policy_paraphrase", "score_correctness")])
@title("Faithfulness and correctness controls are scored by their evaluation adapters [{param_id}]")
def test_score_control_uses_adapters(monkeypatch, control_id, target):
    pytest.importorskip("ragas")
    adapter = AsyncMock(return_value={"value": 0.25})
    monkeypatch.setattr(judge_validation, target, adapter)
    case, judge = control(control_id), Mock()

    assert asyncio.run(score_control(case, judge)) == 0.25
    adapter.assert_awaited_once_with(case, judge)


@pytest.mark.parametrize(("control_id", "metric_class"), [("relevant_context_first", "ContextPrecisionWithReference"),
        ("notice_fact_not_retrieved", "ContextRecall")])
@title(
        "Context controls are scored by the RAGAS metric with the control's question, reference and contexts [{param_id}]"
)
def test_score_control_uses_context_metrics(monkeypatch, control_id, metric_class):
    collections = pytest.importorskip("ragas.metrics.collections")
    metric = Mock(ascore=AsyncMock(return_value=Mock(value=0.25)))
    constructor = Mock(return_value=metric)
    monkeypatch.setattr(collections, metric_class, constructor)
    case, judge = control(control_id), Mock()

    assert asyncio.run(score_control(case, judge)) == 0.25
    constructor.assert_called_once_with(llm=judge)
    metric.ascore.assert_awaited_once_with(
            user_input=case["user_input"], reference=case["reference"], retrieved_contexts=case["retrieved_contexts"])


@pytest.mark.parametrize(("statuses", "status"), [(["matched", "matched"], "matched"),
        (["matched", "mismatch"], "mismatch"), (["mismatch", "error", "matched"], "error")],
        ids=["matched", "mismatch", "error-wins"])
@title("The validation summary counts outcomes and never claims release suitability [{param_id}]")
def test_summary(statuses, status):
    results = [{"status": s, "judge_calls": [{}] * 2} for s in statuses]

    summary = summarize_validation(results)

    assert summary["status"] == status
    assert summary["counts"] == {s: statuses.count(s) for s in ("matched", "mismatch", "error")}
    assert summary["observed_judge_calls"] == 2 * len(statuses)
    assert (summary["release_suitability"], summary["human_review"],
            summary["threshold_decision"]) == ("not_established", "pending", "retain_experimental_thresholds")


@pytest.mark.parametrize("results", [[], [{"status": "skipped", "judge_calls": []}]], ids=["empty", "unknown-status"])
@title("An empty experiment or an unknown outcome cannot be summarised [{param_id}]")
def test_summary_rejects_invalid_results(results):
    with pytest.raises(ValueError, match="nonempty results with known statuses"):
        summarize_validation(results)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """The judge-validation CLI run from the automation directory, with Ollama and the batch replaced."""
    pytest.importorskip("ragas")
    monkeypatch.chdir(AUTOMATION_ROOT)
    monkeypatch.delenv("ANYTHINGLLM_API_KEY_FILE", raising=False)
    state = SimpleNamespace(output=tmp_path / "judge.json", transport=Mock(), ollama=Mock())
    state.ollama.list_models.return_value = model_catalog((TEST_MODEL, MODEL_DIGEST))
    state.http_class = Mock(return_value=state.transport)
    state.judge_class = Mock(return_value=Mock(options={"temperature": 0}))
    state.evaluate = AsyncMock(
            return_value=[{
            "id": "receipt_condition_supported",
            "status": "matched",
            "judge_calls": [{}, {}]}])
    monkeypatch.setattr(judge_validation, "HttpClient", state.http_class)
    monkeypatch.setattr(judge_validation, "OllamaClient", Mock(return_value=state.ollama))
    monkeypatch.setattr("llm_testkit.evaluation.ollama_judge.OllamaJudge", state.judge_class)
    monkeypatch.setattr(judge_validation, "evaluate_judge_controls", state.evaluate)

    def invoke(*arguments: str) -> int:
        monkeypatch.setattr("sys.argv", ["judge_validation", "--judge-model", TEST_MODEL, *arguments])
        return main()

    state.invoke = invoke
    state.report = lambda: json.loads(state.output.read_text())
    return state


@title("A dry run prints the plan of eight controls and 18 judge calls without any network access")
def test_cli_dry_run(cli, capsys):
    assert cli.invoke("--dry-run") == 0

    assert capsys.readouterr().out.strip() == (
            "Controls: 8; maximum judge calls: 18; generations: 0; concurrency: 1; retries: 0")
    cli.http_class.assert_not_called()


@title("A live run records the catalog, judge and outcomes, prints each control and exits with 0 when matched")
def test_cli_records_matched_run(cli, capsys):
    exit_code = cli.invoke("--control", "receipt_condition_supported", "--output", str(cli.output))

    report = cli.report()
    catalog = curated_controls()
    assert exit_code == 0
    assert (report["schema_version"], report["status"]) == (1, "matched")
    assert report["response_origin"] == "synthetic_engineering_labelled_controls"
    assert (report["catalog_version"], report["catalog_sha256"]) == (catalog.version, catalog.sha256)
    assert (report["policy_sha256"],
            report["golden_dataset_sha256"]) == (GOLDEN_DATASET.policy_sha256, GOLDEN_DATASET.sha256)
    assert (report["judge_model"], report["judge_model_digest"]) == (TEST_MODEL, MODEL_DIGEST)
    assert report["ragas_version"] == version("ragas")
    assert datetime.fromisoformat(report["created_at"]).utcoffset() == timedelta(0)
    assert (report["maximum_judge_calls"],
            report["execution"]) == (2, {
            "concurrency": 1,
            "retries": 0,
            "generations": 0})
    assert "not statistical calibration or application acceptance" in report["scope"]
    assert report["judge_configuration"] == {"options": {"temperature": 0}, "think": False, "retries": 0}
    assert (report["counts"], report["observed_judge_calls"]) == ({"matched": 1, "mismatch": 0, "error": 0}, 2)
    output = capsys.readouterr().out
    assert "receipt_condition_supported: matched" in output
    assert f"Judge validation: matched; evidence saved to {cli.output}" in output
    cli.transport.close.assert_called_once()


@title("The live run creates each judge with its control's budget")
def test_cli_judge_factory_uses_budgets(cli):
    cli.invoke("--control", "manager_policy_paraphrase", "--output", str(cli.output))

    cases, factory = cli.evaluate.await_args.args
    factory(4)

    assert [case["id"] for case in cases] == ["manager_policy_paraphrase"]
    settings = Settings()
    assert cli.judge_class.call_args_list == [
            call(cli.ollama, TEST_MODEL, settings.llm_timeout, max_calls=1),
            call(cli.ollama, TEST_MODEL, settings.llm_timeout, max_calls=4)]
    cli.http_class.assert_called_once_with(settings.ollama_base_url, settings.http_timeout)


@title("A mismatched control makes the run exit with 1")
def test_cli_reports_mismatch(cli):
    cli.evaluate.return_value = [{"id": "receipt_condition_supported", "status": "mismatch", "judge_calls": []}]

    assert cli.invoke("--control", "receipt_condition_supported", "--output", str(cli.output)) == 1
    assert cli.report()["status"] == "mismatch"


@pytest.mark.parametrize(
        "inventory", [
        pytest.param(lambda: model_catalog(("other-model", MODEL_DIGEST)), id="model-not-installed"),
        pytest.param(lambda: model_catalog(status_code=503), id="inventory-error")])
@title("A judge model that cannot be verified is saved as an error and the transport is closed [{param_id}]")
def test_cli_saves_inventory_error(cli, inventory):
    cli.ollama.list_models.return_value = inventory()

    assert cli.invoke("--control", "receipt_condition_supported", "--output", str(cli.output)) == 1

    report = cli.report()
    assert (report["status"], report["results"], report["error"]["type"]) == ("error", [], "AssertionError")
    cli.evaluate.assert_not_awaited()
    cli.transport.close.assert_called_once()


@pytest.mark.parametrize(("arguments", "message"), [
        pytest.param([], "Live judge validation requires --output", id="live-without-output"),
        pytest.param(["--control", "unknown", "--dry-run"], "nonempty, known judge control ids", id="unknown-control"),
        pytest.param(["--max-judge-calls", "17", "--dry-run"], "exceed the explicit call budget", id="over-budget"),
        pytest.param(["--controls", "missing.json", "--dry-run"], "No such file", id="missing-catalog")])
@title("The CLI refuses a run without output, an invalid selection or an unreadable catalog [{param_id}]")
def test_cli_refuses_invalid_invocation(cli, capsys, arguments, message):
    with pytest.raises(SystemExit) as stopped:
        cli.invoke(*arguments)

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err
    cli.http_class.assert_not_called()


@title("The CLI refuses to overwrite saved evidence before any network access")
def test_cli_refuses_existing_output(cli, capsys):
    cli.output.write_text("original")

    with pytest.raises(SystemExit) as stopped:
        cli.invoke("--output", str(cli.output))

    assert stopped.value.code == 2
    assert "Output already exists" in capsys.readouterr().err
    assert cli.output.read_text() == "original"
    cli.http_class.assert_not_called()


@title("The default catalog is the reviewed judge-validation catalog")
def test_cli_default_catalog_is_reviewed_catalog(cli, capsys):
    cli.invoke("--dry-run", "--control", "receipt_condition_supported", "--controls", str(JUDGE_CONTROLS_FILE))
    explicit = capsys.readouterr().out

    cli.invoke("--dry-run", "--control", "receipt_condition_supported")

    assert capsys.readouterr().out == explicit
