"""Reference-based factual correctness: RAGAS scoring, evidence integrity, labelled controls and the CLI.

Every rejection names the rule that must fire, so a check that fails for an unrelated reason cannot pass.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import ANY, AsyncMock, Mock

import pytest

from llm_testkit.config import Settings
from llm_testkit.evaluation.correctness import (
        METRIC_CONFIGURATION, bind_case, check_control, check_correctness_evidence, evaluate_correctness_report, main,
        result_from_calls, score_correctness, validate_control, validate_result)
from llm_testkit.reporting.quality import build_quality_report
from llm_testkit.reporting.steps import title
from test_support.builders.correctness import (
        CONTROLS_FILE, judge_calls, judge_outputs, labelled_claims, make_correctness_evidence,
        make_faithfulness_evidence, make_result)
from test_support.builders.golden import (
        GOLDEN_DATASET, GOLDEN_DATASET_FILE, PAID_LEAVE, POLICY_FILE, TEST_DATA, make_paid_leave_sample)
from test_support.builders.ollama import chat_response, model_catalog
from test_support.builders.optional import load_ollama_judge
from test_support.data.common import MODEL_DIGEST, TEST_MODEL
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit

SCORE = "llm_testkit.evaluation.correctness.score_correctness"
LEAVE_ENTITLEMENT = "Each employee receives 23 working days of paid leave."
NOTICE_PERIOD = "A request needs 12 calendar days before leave starts."
MANAGER_APPROVAL = "The direct manager approves the request."


@pytest.fixture
def controls() -> list[dict]:
    """A fresh copy of the curated correctness controls for each test."""
    return json.loads(CONTROLS_FILE.read_text())["cases"]


def control(controls: list[dict], control_id: str) -> dict:
    return next(case for case in controls if case["id"] == control_id)


def with_supported_answer_claims(*claims: str):
    """Replace the answer claims together with matching 'supported' verdicts."""
    return lambda result: result.update(
            response_claims=list(claims), response_verdicts=labelled_claims(claims, [1] * len(claims)))


def scripted_judge(result: dict) -> tuple[Any, Mock]:
    """A real OllamaJudge with a four-call budget whose transport replays the judge outputs for the result."""
    client = Mock()
    client.structured_chat.side_effect = [chat_response(output) for output in judge_outputs(result)]
    return load_ollama_judge()(client, TEST_MODEL, max_calls=4), client


# Factual F1 = 2TP / (2TP + FP + FN). Response labels give TP/FP, reference labels give FN.


@pytest.mark.parametrize(("response_labels", "reference_labels", "expected_f1"), [
        pytest.param((1, 1), (1, 1), 1.0, id="correct"),
        pytest.param((1, ), (1, 0), 0.67, id="incomplete"),
        pytest.param((0, 0), (0, 0), 0.0, id="contradicted"),
        pytest.param((1, 1, 0), (1, 1), 0.8, id="extra-claim")])
@title("Real RAGAS factual F1 distinguishes correct, incomplete and incorrect evidence [{param_id}]")
def test_ragas_factual_f1_follows_judge_verdicts(response_labels, reference_labels, expected_f1):
    judge, client = scripted_judge(make_result(response_labels, reference_labels))

    result = asyncio.run(score_correctness(make_paid_leave_sample(), judge))

    assert result["value"] == expected_f1
    assert client.structured_chat.call_count == 4
    with pytest.raises(ValueError, match="budget"):
        judge.generate("An additional request must be rejected", object)
    assert client.structured_chat.call_count == 4, "Budget rejection must happen before another transport call"


@title("Scoring configures RAGAS with the same metric configuration that evidence records")
def test_scoring_uses_recorded_metric_configuration(monkeypatch):
    judge, _ = scripted_judge(make_result())
    collections = pytest.importorskip("ragas.metrics.collections")
    metric = Mock(wraps=collections.FactualCorrectness)
    monkeypatch.setattr(collections, "FactualCorrectness", metric)

    asyncio.run(score_correctness(make_paid_leave_sample(), judge))

    metric.assert_called_once_with(llm=judge, **METRIC_CONFIGURATION)


@title("Consistent judge evidence is accepted and summarised as TP/FP/FN counts")
def test_validate_result_derives_counts_from_verdicts():
    result = make_result(response_labels=(1, 0), reference_labels=(1, 0), value=0.5)

    validated = validate_result(result)

    assert validated["counts"] == {"tp": 1, "fp": 1, "fn": 1}
    assert validated["value"] == 0.5


@pytest.mark.parametrize(("corrupt", "error"), [
        pytest.param(
        lambda result: result["reference_verdicts"].pop(), "Incomplete or duplicate reference claim",
        id="unverified-reference-claim"),
        pytest.param(
        lambda result: result.update(response_claims=[]), "Incomplete or duplicate response claim",
        id="no-response-claims"),
        pytest.param(
        lambda result: result.update(response_claims=tuple(result["response_claims"])),
        "Incomplete or duplicate response claim", id="claims-not-a-list"),
        pytest.param(with_supported_answer_claims(" "), "Incomplete or duplicate response claim", id="blank-claim"),
        pytest.param(
        with_supported_answer_claims("Same claim", "Same claim"), "Incomplete or duplicate response claim",
        id="duplicate-claim"),
        pytest.param(
        lambda result: result["response_verdicts"][0].update(verdict=True), "zero or one", id="boolean-verdict"),
        pytest.param(
        lambda result: result["response_verdicts"][0].update(verdict=2), "zero or one", id="verdict-out-of-range"),
        pytest.param(
        lambda result: result["reference_verdicts"][0].update(reason=" "), "requires a reason", id="blank-reason"),
        pytest.param(lambda result: result.update(value=0.5), "F1 does not match", id="score-differs-from-verdicts"),
        pytest.param(
        lambda result: result.update(counts=dict(tp=9, fp=0, fn=0)), "counts do not match",
        id="counts-differ-from-verdicts")])
@title("Judge evidence that is incomplete, duplicated or inconsistent with its score is rejected [{param_id}]")
def test_validate_result_rejects_inconsistent_evidence(corrupt, error):
    result = make_result()
    corrupt(result)

    with pytest.raises(ValueError, match=error):
        validate_result(result)


@title("Saved judge calls rebuild the result only from two decompositions and two verifications")
def test_result_from_calls_requires_four_calls():
    calls = judge_calls(make_result())

    assert result_from_calls(calls, 1.0)["counts"] == {"tp": 2, "fp": 0, "fn": 0}
    with pytest.raises(ValueError, match="two decompositions and two claim verifications"):
        result_from_calls(calls[:3], 1.0)


@title("A captured sample binds to its golden case when question, reference and provenance match")
def test_bind_case_accepts_matching_sample():
    sample = make_paid_leave_sample()
    sample["metadata"] = {
            "golden_case_id": PAID_LEAVE.id,
            "golden_dataset_sha256": GOLDEN_DATASET.sha256,
            "policy_sha256": GOLDEN_DATASET.policy_sha256}

    assert bind_case(sample, GOLDEN_DATASET, PAID_LEAVE.id) == PAID_LEAVE


@pytest.mark.parametrize(("corrupt", "error"), [
        pytest.param(lambda sample: sample.update(reference="Edited"), "question/reference", id="edited-reference"),
        pytest.param(lambda sample: sample.update(user_input="Edited"), "question/reference", id="edited-question"),
        pytest.param(
        lambda sample: sample.update(metadata={"golden_case_id": "other"}), "golden_case_id does not match",
        id="other-case"),
        pytest.param(
        lambda sample: sample.update(metadata={"golden_dataset_sha256": "stale"}),
        "golden_dataset_sha256 does not match", id="stale-dataset"),
        pytest.param(
        lambda sample: sample.update(metadata={"policy_sha256": "stale"}), "policy_sha256 does not match",
        id="stale-policy")])
@title("Samples with edited inputs or stale golden provenance are rejected [{param_id}]")
def test_bind_case_rejects_unbound_sample(corrupt, error):
    sample = make_paid_leave_sample()
    corrupt(sample)

    with pytest.raises(ValueError, match=error):
        bind_case(sample, GOLDEN_DATASET, PAID_LEAVE.id)


@title("Binding to a case that is not in the golden dataset is rejected")
def test_bind_case_rejects_unknown_case():
    with pytest.raises(ValueError, match="Unknown golden case"):
        bind_case(make_paid_leave_sample(), GOLDEN_DATASET, "not_in_dataset")


@title("Saved correctness evidence is accepted when bound to its sample, case and raw judge calls")
def test_evidence_check_returns_validated_measurement():
    sample = make_paid_leave_sample()

    checked = check_correctness_evidence(make_correctness_evidence(sample), "sample", sample, GOLDEN_DATASET)

    assert checked["value"] == 1.0
    assert checked["counts"] == {"tp": 2, "fp": 0, "fn": 0}
    assert checked["threshold"] is None
    assert checked["metric_configuration"] == METRIC_CONFIGURATION
    assert (checked["judge_model"], checked["judge_model_digest"]) == (TEST_MODEL, MODEL_DIGEST)
    assert (checked["golden_case_id"], checked["golden_dataset_sha256"]) == (PAID_LEAVE.id, GOLDEN_DATASET.sha256)


@pytest.mark.parametrize(("changes", "error"), [
        pytest.param({"schema_version": 2}, "Unsupported correctness report", id="schema-version"),
        pytest.param({"metric": "faithfulness"}, "Unsupported correctness report", id="other-metric"),
        pytest.param({"status": "error"}, "completed application evidence", id="unfinished"),
        pytest.param({"response_origin": "synthetic_control"}, "completed application evidence",
        id="synthetic-control"),
        pytest.param({"sample_sha256": "other"}, "checksum mismatch", id="other-sample"),
        pytest.param({"golden_dataset_sha256": "stale"}, "checksum mismatch", id="stale-dataset"),
        pytest.param({"golden_case_id": "not_in_dataset"}, "Unknown golden case", id="unknown-case"),
        pytest.param({"metric_configuration": {
        **METRIC_CONFIGURATION, "beta": 2.0}}, "inputs/configuration mismatch", id="metric-configuration"),
        pytest.param({"question": "Edited"}, "inputs/configuration mismatch", id="question"),
        pytest.param({"reference": "Edited"}, "inputs/configuration mismatch", id="reference"),
        pytest.param({"reference_sha256": "edited"}, "inputs/configuration mismatch", id="reference-checksum"),
        pytest.param({"response": "Edited"}, "inputs/configuration mismatch", id="response"),
        pytest.param({"result": make_result(value=0.5)}, "F1 does not match", id="inconsistent-summary"),
        pytest.param({"judge_calls": []}, "two decompositions and two claim verifications", id="missing-raw-calls")])
@title("Correctness evidence that is unfinished, synthetic or bound to other inputs is rejected [{param_id}]")
def test_evidence_check_rejects_unbound_evidence(changes, error):
    sample = make_paid_leave_sample()
    evidence = {**make_correctness_evidence(sample), **changes}

    with pytest.raises(ValueError, match=error):
        check_correctness_evidence(evidence, "sample", sample, GOLDEN_DATASET)


@title("A summary that differs from internally consistent raw judge calls is rejected")
def test_evidence_check_rejects_summary_that_differs_from_raw_calls():
    sample = make_paid_leave_sample()
    evidence = make_correctness_evidence(sample)
    # The raw calls are valid on their own, so only the summary comparison can detect the substitution.
    evidence["judge_calls"] = judge_calls(make_result(reference_claims=["Other claim 0", "Other claim 1"]))

    with pytest.raises(ValueError, match="differs from saved judge calls"):
        check_correctness_evidence(evidence, "sample", sample, GOLDEN_DATASET)


@title("All curated correctness controls have valid labelled expectations for the paid-leave case")
def test_curated_controls_are_valid(controls):
    assert sorted(case["id"] for case in controls) == ["contradicted", "correct", "incomplete", "paraphrase"]
    for case in controls:
        validate_control(case)
        assert case["case_id"] == PAID_LEAVE.id


@pytest.mark.parametrize(("corrupt", "error"), [
        pytest.param(lambda case: case.update(id=""), "nonempty id", id="blank-id"),
        pytest.param(lambda case: case.pop("case_id"), "nonempty case_id", id="missing-case"),
        pytest.param(lambda case: case.update(response=" "), "nonempty response", id="blank-response"),
        pytest.param(lambda case: case.update(expected_f1_range=[0.5]), "requires an F1 range", id="single-bound"),
        pytest.param(lambda case: case.update(expected_f1_range=[0.9, 0.1]), "Invalid control F1", id="reversed-range"),
        pytest.param(
        lambda case: case.update(response_verdict=True), "Invalid control response verdict",
        id="boolean-response-verdict"),
        pytest.param(
        lambda case: case.update(response_verdict=2), "Invalid control response verdict",
        id="response-verdict-out-of-range"),
        pytest.param(lambda case: case.update(reference_rules={}), "hand-labelled reference rules", id="no-rules"),
        pytest.param(
        lambda case: case["reference_rules"]["notice period"].update(verdict=True), "Invalid control reference verdict",
        id="boolean-rule-verdict"),
        pytest.param(
        lambda case: case["reference_rules"]["notice period"].update(verdict=2), "Invalid control reference verdict",
        id="rule-verdict-out-of-range"),
        pytest.param(
        lambda case: case["reference_rules"]["notice period"].update(pattern=""), "Missing control reference pattern",
        id="blank-pattern"),
        pytest.param(
        lambda case: case["reference_rules"]["notice period"].update(pattern=".*"), "must not match empty text",
        id="pattern-matches-anything"),
        pytest.param(
        lambda case: case["reference_rules"]["notice period"].update(pattern="^$"), "must not match empty text",
        id="pattern-matches-only-empty-text")])
@title("Malformed correctness controls are rejected before any judge call [{param_id}]")
def test_validate_control_rejects_malformed_control(controls, corrupt, error):
    incomplete = control(controls, "incomplete")
    corrupt(incomplete)

    with pytest.raises(ValueError, match=error):
        validate_control(incomplete)


@title("Controls check claim semantics, so any decomposition with matching labels is accepted")
def test_check_control_accepts_decomposition_with_matching_labels(controls):
    # Three reference claims instead of two; rule patterns match regardless of letter case.
    result = make_result(
            response_labels=(1, 1), reference_labels=(1, 1, 0), value=0.8,
            reference_claims=[LEAVE_ENTITLEMENT, "Paid leave is 23 WORKING DAYS per year.", NOTICE_PERIOD])

    check_control(result, control(controls, "incomplete"))


@title("A control accepts a score exactly on its F1 bounds")
def test_check_control_accepts_score_on_range_bounds(controls):
    result = make_result(reference_claims=[LEAVE_ENTITLEMENT, NOTICE_PERIOD])

    check_control(result, control(controls, "correct"))


@pytest.mark.parametrize(("response_labels", "reference_labels", "f1", "reference_claims", "error"), [
        pytest.param((1, 1),
        (1, 1), 1.0, [LEAVE_ENTITLEMENT, NOTICE_PERIOD], "outside its expected range", id="score-out-of-range"),
        pytest.param((1, 0),
        (1, 0), 0.5, [LEAVE_ENTITLEMENT, NOTICE_PERIOD], "response verdict disagrees", id="unsupported-answer-claim"),
        pytest.param((1, 1), (0, 1), 0.8, [LEAVE_ENTITLEMENT, NOTICE_PERIOD], "label: leave entitlement",
        id="covered-claim-labelled-missing"),
        pytest.param((1, 1),
        (1, 0), 0.8, [LEAVE_ENTITLEMENT, MANAGER_APPROVAL], "label: notice period", id="rule-without-claim"),
        pytest.param((1, 1), (1, 0, 1), 0.8, [LEAVE_ENTITLEMENT, NOTICE_PERIOD, MANAGER_APPROVAL],
        "unlabelled reference claims", id="claim-without-rule")])
@title("Judge labels that disagree with the incomplete-answer control are rejected [{param_id}]")
def test_check_control_rejects_disagreeing_labels(
        controls, response_labels, reference_labels, f1, reference_claims, error):
    result = make_result(response_labels, reference_labels, f1, reference_claims=reference_claims)

    with pytest.raises(ValueError, match=error):
        check_control(result, control(controls, "incomplete"))


@pytest.fixture
def saved_evidence(tmp_path):
    """A captured sample with faithfulness and low-score correctness evidence saved next to it."""
    sample = make_paid_leave_sample()
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(sample))
    checksum = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    faithfulness_path = tmp_path / "faithfulness.json"
    faithfulness_path.write_text(json.dumps(make_faithfulness_evidence(checksum)))
    correctness_path = tmp_path / "correctness.json"
    low_score = make_result(response_labels=(0, 0), reference_labels=(0, 0), value=0.0)
    correctness_path.write_text(json.dumps(make_correctness_evidence(sample, sample_sha256=checksum, result=low_score)))
    return SimpleNamespace(sample=sample_path, faithfulness=faithfulness_path, correctness=correctness_path)


def quality_report(paths: SimpleNamespace) -> dict:
    return build_quality_report(
            paths.sample, paths.faithfulness, TEST_DATA / "quality-paid-leave.json", correctness_path=paths.correctness)


def dimension(report: dict, metric: str) -> dict:
    return next(item for item in report["dimensions"] if item.get("metric") == metric)


@title("A valid low correctness score is recorded as a measurement, not a failure")
def test_quality_report_records_low_correctness_as_measurement(saved_evidence):
    report = quality_report(saved_evidence)

    assert report["status"] == "checks_passed"
    assert len(report["dimensions"]) == 4
    assert dimension(report, "factual_correctness")["status"] == "measured"
    assert dimension(report, "factual_correctness")["details"]["value"] == 0.0


@title("Broken correctness evidence is an error of its own dimension and leaves faithfulness measured")
def test_quality_report_isolates_broken_correctness_evidence(saved_evidence):
    evidence = json.loads(saved_evidence.correctness.read_text())
    evidence["sample_sha256"] = "other"
    saved_evidence.correctness.write_text(json.dumps(evidence))

    report = quality_report(saved_evidence)

    assert report["status"] == "error"
    assert dimension(report, "factual_correctness")["status"] == "error"
    assert dimension(report, "faithfulness")["status"] == "measured"


@pytest.fixture
def offline_service(tmp_path, monkeypatch):
    """evaluate_correctness_report with the Ollama transport, model catalog and judge replaced by doubles."""
    pytest.importorskip("ragas")
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps(make_paid_leave_sample()))
    service = SimpleNamespace(sample_path=sample_path, transport=Mock(), ollama=Mock(), judge=Mock())
    service.ollama.list_models.return_value = model_catalog((TEST_MODEL, MODEL_DIGEST))
    service.judge.configure_mock(calls=[{"output": "raw judge call"}], options={"temperature": 0})
    service.http_class = Mock(return_value=service.transport)
    service.ollama_class = Mock(return_value=service.ollama)
    service.judge_class = Mock(return_value=service.judge)
    monkeypatch.setattr("llm_testkit.evaluation.correctness.HttpClient", service.http_class)
    monkeypatch.setattr("llm_testkit.evaluation.correctness.OllamaClient", service.ollama_class)
    monkeypatch.setattr("llm_testkit.evaluation.ollama_judge.OllamaJudge", service.judge_class)
    return service


def evaluate(service: SimpleNamespace, *, control: dict | None = None) -> dict:
    return evaluate_correctness_report(
            service.sample_path, dataset_path=GOLDEN_DATASET_FILE, policy_file=POLICY_FILE, case_id=PAID_LEAVE.id,
            settings=Settings(), judge_model=TEST_MODEL, control=control)


@title("A completed report records the measurement, its provenance and the judge, then closes the transport")
def test_report_records_completed_measurement(offline_service, monkeypatch):
    score = AsyncMock(return_value=make_result())
    monkeypatch.setattr(SCORE, score)
    settings = Settings()
    expected_judge_configuration = dict(options=offline_service.judge.options, think=False, retries=0, max_calls=4)

    report = evaluate(offline_service)

    assert (report["schema_version"], report["metric"], report["status"]) == (1, "factual_correctness", "completed")
    assert report["result"]["value"] == 1.0
    assert (report["threshold"], report["metric_configuration"]) == (None, METRIC_CONFIGURATION)
    assert "not a calibrated quality gate" in report["interpretation"]
    assert report["ragas_version"] == version("ragas")
    assert datetime.fromisoformat(report["created_at"]).utcoffset() == timedelta(0)
    assert report["sample_sha256"] == hashlib.sha256(offline_service.sample_path.read_bytes()).hexdigest()
    assert (report["golden_case_id"], report["golden_dataset_sha256"]) == (PAID_LEAVE.id, GOLDEN_DATASET.sha256)
    assert report["golden_dataset_version"] == GOLDEN_DATASET.version
    assert report["reference_sha256"] == hashlib.sha256(PAID_LEAVE.reference.encode()).hexdigest()
    assert (report["question"], report["reference"]) == (PAID_LEAVE.question, PAID_LEAVE.reference)
    assert (report["response_origin"], report["response"]) == ("application_sample", PAID_LEAVE.reference)
    assert (report["generation_model"], report["same_generation_and_judge_model"]) == (TEST_MODEL, True)
    assert (report["judge_model"], report["judge_model_digest"]) == (TEST_MODEL, MODEL_DIGEST)
    assert report["judge_configuration"] == expected_judge_configuration
    assert report["judge_calls"] == offline_service.judge.calls
    offline_service.http_class.assert_called_once_with(settings.ollama_base_url, settings.http_timeout)
    offline_service.ollama_class.assert_called_once_with(offline_service.transport)
    offline_service.judge_class.assert_called_once_with(
            offline_service.ollama, TEST_MODEL, settings.llm_timeout, max_calls=4)
    score.assert_awaited_once_with(ANY, offline_service.judge)
    offline_service.transport.close.assert_called_once()


@title("A judge failure is reported with its error and raw calls, and the transport is still closed")
def test_report_preserves_judge_failure(offline_service, monkeypatch):
    monkeypatch.setattr(SCORE, AsyncMock(side_effect=ValueError("Truncated judge response")))

    report = evaluate(offline_service)

    assert report["status"] == "error"
    assert report["error"] == {"type": "ValueError", "message": "Truncated judge response"}
    assert report["judge_calls"] == offline_service.judge.calls
    offline_service.transport.close.assert_called_once()


@title("A judge model missing from the Ollama catalog is an error before the judge is created")
def test_report_requires_installed_judge_model(offline_service, monkeypatch):
    offline_service.ollama.list_models.return_value = model_catalog(("other-model", MODEL_DIGEST))
    monkeypatch.setattr(SCORE, AsyncMock(return_value=make_result()))

    report = evaluate(offline_service)

    assert report["status"] == "error"
    assert "Expected installed Ollama model" in report["error"]["message"]
    offline_service.judge_class.assert_not_called()
    offline_service.transport.close.assert_called_once()


@title("A synthetic control replaces the answer and is matched against its labels")
def test_report_evaluates_matching_control(offline_service, monkeypatch, controls):
    incomplete = control(controls, "incomplete")
    labelled_as_incomplete = make_result(
            response_labels=(1, 1), reference_labels=(1, 0), value=0.8,
            reference_claims=[LEAVE_ENTITLEMENT, NOTICE_PERIOD])
    score = AsyncMock(return_value=labelled_as_incomplete)
    monkeypatch.setattr(SCORE, score)

    report = evaluate(offline_service, control=incomplete)

    assert (report["status"], report["control_status"]) == ("completed", "matched")
    assert (report["response_origin"], report["control"]) == ("synthetic_control", incomplete)
    assert report["response"] == incomplete["response"]
    assert score.await_args.args[0]["response"] == incomplete["response"]


@title("A control whose judge labels disagree is completed but marked as a mismatch")
def test_report_marks_control_mismatch(offline_service, monkeypatch, controls):
    monkeypatch.setattr(SCORE, AsyncMock(return_value=make_result()))

    report = evaluate(offline_service, control=control(controls, "incomplete"))

    assert (report["status"], report["control_status"]) == ("completed", "mismatch")
    assert "outside its expected range" in report["control_error"]


@title("A control for another golden case is rejected before the transport and judge are created")
def test_report_rejects_control_for_other_case(offline_service, monkeypatch, controls):
    other_case = {**control(controls, "incomplete"), "case_id": "carryover_limit"}
    monkeypatch.setattr(SCORE, AsyncMock(return_value=make_result()))

    report = evaluate(offline_service, control=other_case)

    assert report["status"] == "error"
    assert report["error"]["message"] == "Control belongs to a different golden case"
    assert "judge_calls" not in report
    offline_service.transport.close.assert_not_called()


@pytest.fixture
def cli_evaluation(monkeypatch):
    """The CLI's evaluation step, replaced so that argument handling and exit codes are checked offline."""
    monkeypatch.delenv("ANYTHINGLLM_API_KEY_FILE", raising=False)
    evaluation = Mock(return_value={"status": "completed"})
    monkeypatch.setattr("llm_testkit.evaluation.correctness.evaluate_correctness_report", evaluation)
    return evaluation


@pytest.fixture
def run_cli(monkeypatch, cli_evaluation):
    def run(*arguments: str) -> int:
        monkeypatch.setattr("sys.argv", ["correctness", "sample.json", "--case", PAID_LEAVE.id, *arguments])
        return main()

    return run


@title("The CLI evaluates the selected case with default inputs and writes the report to a new file")
def test_cli_writes_completed_report(run_cli, cli_evaluation, tmp_path, capsys):
    output = tmp_path / "correctness.json"

    exit_code = run_cli("--output", str(output))

    assert exit_code == 0
    assert json.loads(output.read_text()) == {"status": "completed"}
    assert "Correctness: completed" in capsys.readouterr().out
    cli_evaluation.assert_called_once_with(
            Path("sample.json"), dataset_path=Path("test_data/golden-policy.json"),
            policy_file=Path("test_data/company-policy.txt"), case_id=PAID_LEAVE.id, settings=ANY,
            judge_model="qwen3.5:4b", control=None)
    assert isinstance(cli_evaluation.call_args.kwargs["settings"], Settings)


@title("The CLI passes the chosen inputs, judge model and a control from the default catalog")
def test_cli_passes_selected_inputs(run_cli, cli_evaluation, tmp_path, monkeypatch, controls):
    monkeypatch.chdir(AUTOMATION_ROOT)

    exit_code = run_cli(
            "--output", str(tmp_path / "out.json"), "--dataset", "dataset.json", "--policy", "policy.txt",
            "--judge-model", "judge-model", "--control", "incomplete")

    assert exit_code == 0
    arguments = cli_evaluation.call_args.kwargs
    assert (arguments["dataset_path"], arguments["policy_file"]) == (Path("dataset.json"), Path("policy.txt"))
    assert arguments["judge_model"] == "judge-model"
    assert arguments["control"] == control(controls, "incomplete")


@pytest.mark.parametrize(
        "arguments", [
        pytest.param(["sample.json", "--output", "out.json"], id="without-case"),
        pytest.param(["sample.json", "--case", "paid_leave"], id="without-output")])
@title("The CLI requires the golden case and an output file [{param_id}]")
def test_cli_requires_case_and_output(cli_evaluation, monkeypatch, tmp_path, arguments):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["correctness", *arguments])

    with pytest.raises(SystemExit) as stopped:
        main()

    assert stopped.value.code == 2
    cli_evaluation.assert_not_called()


@pytest.mark.parametrize(
        "report", [dict(status="error"), dict(status="completed", control_status="mismatch")],
        ids=["evaluation-error", "control-mismatch"])
@title("The CLI exits with 1 when evaluation fails or a control does not match [{param_id}]")
def test_cli_exit_code_reports_failure(run_cli, cli_evaluation, tmp_path, report):
    cli_evaluation.return_value = report

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


@title("The CLI refuses to run without the evaluation dependencies")
def test_cli_requires_evaluation_dependencies(run_cli, cli_evaluation, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("llm_testkit.evaluation.correctness.find_spec", lambda name: None)

    with pytest.raises(SystemExit) as stopped:
        run_cli("--output", str(tmp_path / "out.json"))

    assert stopped.value.code == 2
    assert "requirements-evaluation.lock" in capsys.readouterr().err
    cli_evaluation.assert_not_called()


@pytest.mark.parametrize(("catalog", "control_id"), [
        pytest.param({
        "schema_version": 1,
        "cases": [{
        "id": "incomplete"}]}, "unknown", id="unknown-control"),
        pytest.param({
        "schema_version": 1,
        "cases": [{
        "id": "incomplete"}, {
        "id": "incomplete"}]}, "incomplete", id="duplicate-control"),
        pytest.param({
        "schema_version": 2,
        "cases": [{
        "id": "incomplete"}]}, "incomplete", id="unsupported-catalog")])
@title("The CLI refuses an unknown, duplicate or unsupported control before evaluation [{param_id}]")
def test_cli_refuses_unresolvable_control(run_cli, cli_evaluation, tmp_path, capsys, catalog, control_id):
    catalog_path = tmp_path / "controls.json"
    catalog_path.write_text(json.dumps(catalog))

    with pytest.raises(SystemExit) as stopped:
        run_cli("--output", str(tmp_path / "out.json"), "--control", control_id, "--controls", str(catalog_path))

    assert stopped.value.code == 2
    assert "Unknown or duplicate correctness control" in capsys.readouterr().err
    cli_evaluation.assert_not_called()
