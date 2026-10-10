"""Benchmark case evaluation: sample provenance, answer and source checks, metric evidence and the case pipeline."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from llm_testkit.config import Settings
from llm_testkit.evaluation import benchmark as evaluation
from llm_testkit.reporting.steps import title
from test_support.builders.benchmark import (
        GENERATION_DIGEST, JUDGE_DIGEST, MINIMA, MODEL, make_benchmark_sample, make_metric_evidence,
        patch_metric_reports)
from test_support.builders.golden import CASES, GOLDEN_DATASET, GOLDEN_DATASET_FILE, PAID_LEAVE, POLICY_FILE

pytestmark = pytest.mark.unit

METRICS_IN_ORDER = ["faithfulness", "factual_correctness", "context_precision", "context_recall"]


def save(path, sample: dict):
    """Write a sample as plain JSON, which, unlike the evidence writer, can hold NaN or infinite durations."""
    path.write_text(json.dumps(sample))
    return path


@pytest.fixture
def sample_path(tmp_path):
    return save(tmp_path / "sample.json", make_benchmark_sample(PAID_LEAVE))


def checked_dimensions(name: str, evidence: dict, sample_path, *, minima=MINIMA, judge_digest=JUDGE_DIGEST) -> list:
    checksum = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    sample = json.loads(sample_path.read_text())
    return evaluation.metric_dimensions(name, evidence, checksum, sample, GOLDEN_DATASET, MODEL, judge_digest, minima)


@pytest.mark.parametrize(("name", "expected"), [
        pytest.param("faithfulness", [("faithfulness", 1.0, 0.9)], id="faithfulness"),
        pytest.param("correctness", [("factual_correctness", 1.0, 0.8)], id="correctness"),
        pytest.param(
        "relevance", [("context_precision", 1 / (1 + 1e-10), 0.8), ("context_recall", 1.0, 0.9)], id="relevance")])
@title("Valid judge evidence becomes passed metric dimensions with their recorded thresholds [{param_id}]")
def test_metric_dimensions_from_valid_evidence(sample_path, name, expected):
    evidence = make_metric_evidence(PAID_LEAVE, sample_path)[name]

    dimensions = checked_dimensions(name, evidence, sample_path)

    assert dimensions == [{
            "name": metric,
            "metric": metric,
            "value": value,
            "minimum": minimum,
            "status": "passed"} for metric, value, minimum in expected]


@title("A score equal to its threshold passes; a score just below it fails")
def test_metric_dimensions_threshold_boundary(sample_path):
    evidence = make_metric_evidence(PAID_LEAVE, sample_path)["relevance"]
    # Average precision for one useful context is 1 / (1 + 1e-10), just below 1.0.
    minima = {**MINIMA, "context_precision": 1.0, "context_recall": 1.0}

    dimensions = checked_dimensions("relevance", evidence, sample_path, minima=minima)

    assert [(d["metric"], d["status"]) for d in dimensions] == [("context_precision", "failed"),
            ("context_recall", "passed")]


@pytest.mark.parametrize(("change", "message"), [
        pytest.param({"judge_model": "other"}, "Judge identity changed", id="other-judge-model"),
        pytest.param({"judge_model_digest": "other"}, "Judge identity changed", id="other-judge-weights"),
        pytest.param({
        "status": "error",
        "error": "Truncated"}, "Judge evaluation failed: Truncated", id="unfinished"),
        pytest.param({"sample_sha256": "other"}, "ValueError: Relevance sample/dataset checksum mismatch",
        id="invalid-evidence")])
@title("Evidence from another judge, an unfinished run or invalid evidence becomes error dimensions [{param_id}]")
def test_metric_dimensions_report_errors(sample_path, change, message):
    evidence = {**make_metric_evidence(PAID_LEAVE, sample_path)["relevance"], **change}

    dimensions = checked_dimensions("relevance", evidence, sample_path)

    assert [(d["name"], d["metric"], d["status"]) for d in dimensions] == [
            ("context_precision", "context_precision", "error"), ("context_recall", "context_recall", "error")]
    assert all(message in d["error"] for d in dimensions)
    assert all("value" not in d for d in dimensions)


@title("A sample with matching golden provenance, generation model and settings is accepted")
def test_check_sample_accepts_benchmark_sample(sample_path):
    assert evaluation.check_sample(sample_path, GOLDEN_DATASET, PAID_LEAVE,
            MODEL) == json.loads(sample_path.read_text())


@pytest.mark.parametrize("duration", [12.5, 3, 0.5], ids=["float", "integer", "under-a-second"])
@title("A positive finite answer duration is accepted [{param_id}]")
def test_check_sample_accepts_answer_duration(tmp_path, duration):
    sample = make_benchmark_sample(PAID_LEAVE)
    sample["metadata"]["answer_request_seconds"] = duration

    checked = evaluation.check_sample(save(tmp_path / "s.json", sample), GOLDEN_DATASET, PAID_LEAVE, MODEL)

    assert checked["metadata"]["answer_request_seconds"] == duration


def change_metadata(**changes):
    return lambda sample: sample["metadata"].update(changes)


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(lambda sample: sample.update(metadata={}), "lacks matching golden provenance", id="no-provenance"),
        pytest.param(lambda sample: sample.pop("metadata"), "lacks matching golden provenance", id="no-metadata"),
        pytest.param(change_metadata(golden_category="boundary"), "lacks matching golden provenance", id="category"),
        pytest.param(change_metadata(model_digest=""), "model/digest mismatch", id="no-model-digest"),
        pytest.param(
        lambda sample: sample["metadata"].pop("workspace_configuration"), "lacks workspace configuration",
        id="no-workspace-settings"),
        pytest.param(
        change_metadata(workspace_configuration="Policy only"), "lacks workspace configuration",
        id="workspace-settings-not-an-object"), *[
        pytest.param(change_metadata(answer_request_seconds=value), "finite positive", id=f"duration-{name}")
        for name, value in [("boolean", True), ("zero", 0), ("negative", -1), ("nan",
        float("nan")), ("infinite", float("inf")), ("string", "10")]]])
@title("Samples without golden provenance, model identity, settings or a valid duration are rejected [{param_id}]")
def test_check_sample_rejects_unbound_sample(tmp_path, corrupt, message):
    sample = make_benchmark_sample(PAID_LEAVE)
    corrupt(sample)
    path = save(tmp_path / "sample.json", sample)

    with pytest.raises(ValueError, match=message):
        evaluation.check_sample(path, GOLDEN_DATASET, PAID_LEAVE, MODEL)


@title("A sample generated by another model than planned is rejected")
def test_check_sample_rejects_other_generation_model(sample_path):
    with pytest.raises(ValueError, match="model/digest mismatch"):
        evaluation.check_sample(sample_path, GOLDEN_DATASET, PAID_LEAVE, "qwen2.5:7b")


@title("A reviewed answer cited from the captured policy document passes both acceptance checks")
def test_acceptance_dimensions_pass():
    dimensions = evaluation.acceptance_dimensions(make_benchmark_sample(PAID_LEAVE), PAID_LEAVE)

    assert dimensions == [{
            "name": "Reviewed answer rules",
            "status": "passed"}, {
            "name": "Document sources",
            "status": "passed"}]


@title("An answer missing a required fact fails the answer rules but not the source check")
def test_acceptance_dimensions_fail_incomplete_answer():
    sample = make_benchmark_sample(PAID_LEAVE)
    sample["response"] = "Each employee receives 23 working days of paid leave per year."

    answer, sources = evaluation.acceptance_dimensions(sample, PAID_LEAVE)

    assert (answer["status"], sources["status"]) == ("failed", "passed")
    assert "advance notice" in answer["error"]


@pytest.mark.parametrize(
        "contexts", [
        pytest.param(["Unrelated document"], id="no-policy-document"),
        pytest.param(["sourceDocument: other-policy.txt"], id="unrecognised-title"),
        pytest.param([
        f"sourceDocument: automation-{'a' * 32}-company-policy.txt",
        f"sourceDocument: automation-{'b' * 32}-company-policy.txt"], id="two-policy-documents")])
@title("The source check is an error unless the model saw exactly one captured policy document [{param_id}]")
def test_acceptance_dimensions_require_one_captured_document(contexts):
    sample = {**make_benchmark_sample(PAID_LEAVE), "retrieved_contexts": contexts}

    _, sources = evaluation.acceptance_dimensions(sample, PAID_LEAVE)

    assert sources == {
            "name": "Document sources",
            "status": "error",
            "error": "ValueError: Expected exactly one captured policy document"}


@title("A citation without the reviewed source fragments fails the source check")
def test_acceptance_dimensions_fail_unsupported_citation():
    sample = make_benchmark_sample(PAID_LEAVE)
    sample["response_sources"][0]["text"] = "Unrelated passage."

    _, sources = evaluation.acceptance_dimensions(sample, PAID_LEAVE)

    assert sources["status"] == "failed"
    assert "23 working days" in sources["error"]


@pytest.fixture
def case_run(tmp_path, sample_path, monkeypatch):
    """A saved paid-leave sample with the three judge report services replaced by perfect evidence."""
    directory = tmp_path / "case-001"
    directory.mkdir()
    evidence = make_metric_evidence(PAID_LEAVE, sample_path)
    reports = patch_metric_reports(monkeypatch, evidence)
    return SimpleNamespace(sample_path=sample_path, directory=directory, evidence=evidence, reports=reports)


def evaluate_case(run: SimpleNamespace, *, case=PAID_LEAVE, model=MODEL) -> dict:
    return evaluation.evaluate_case(
            run.sample_path, directory=run.directory, dataset=GOLDEN_DATASET, dataset_path=GOLDEN_DATASET_FILE,
            policy_file=POLICY_FILE, case=case, model=model, judge_model=MODEL, judge_digest=JUDGE_DIGEST,
            settings=Settings(), minima=MINIMA)


@title("A case row records the sample, both checks, four passed metrics, the judge calls and evidence checksums")
def test_evaluate_case_records_all_dimensions(case_run):
    row = evaluate_case(case_run)

    assert (row["case_id"], row["category"], row["model"]) == ("paid_leave", "multi_fact", MODEL)
    assert row["sample_sha256"] == hashlib.sha256(case_run.sample_path.read_bytes()).hexdigest()
    assert (row["model_digest"], row["workspace_configuration"]["openAiPrompt"]) == (GENERATION_DIGEST, "Policy only")
    assert (row["question"], row["reference"],
            row["answer"]) == (PAID_LEAVE.question, PAID_LEAVE.reference, PAID_LEAVE.reference)
    assert [d["name"] for d in row["dimensions"]] == ["Reviewed answer rules", "Document sources", *METRICS_IN_ORDER]
    assert {d["status"] for d in row["dimensions"]} == {"passed"}
    assert row["judge_calls"] == 2 + 4 + 2
    assert "answer_request_seconds" not in row
    for name, evidence in case_run.evidence.items():
        saved = case_run.directory / f"{name}.json"
        assert json.loads(saved.read_text()) == evidence
        assert row["evidence_sha256"][name] == hashlib.sha256(saved.read_bytes()).hexdigest()


@title("Each judge report service receives the sample, the golden inputs, the settings and the judge model")
def test_evaluate_case_calls_report_services(case_run):
    evaluate_case(case_run)

    golden = dict(dataset_path=GOLDEN_DATASET_FILE, policy_file=POLICY_FILE, case_id="paid_leave", judge_model=MODEL)
    case_run.reports["faithfulness"].assert_called_once_with(
            case_run.sample_path, settings=Settings(), judge_model=MODEL)
    case_run.reports["correctness"].assert_called_once_with(case_run.sample_path, settings=Settings(), **golden)
    case_run.reports["relevance"].assert_called_once_with(case_run.sample_path, settings=Settings(), **golden)


@title("A recorded answer duration is copied to the case row")
def test_evaluate_case_keeps_answer_duration(case_run):
    sample = make_benchmark_sample(PAID_LEAVE)
    sample["metadata"]["answer_request_seconds"] = 12.5
    save(case_run.sample_path, sample)

    assert evaluate_case(case_run)["answer_request_seconds"] == 12.5


@pytest.mark.parametrize(("service", "metrics"), [
        pytest.param("faithfulness", ["faithfulness"], id="faithfulness"),
        pytest.param("relevance", ["context_precision", "context_recall"], id="relevance")])
@title(
        "A failing judge service turns only its own metrics into errors; the other metrics are still evaluated [{param_id}]"
)
def test_evaluate_case_isolates_failed_service(case_run, service, metrics):
    case_run.reports[service].side_effect = RuntimeError("Judge offline")

    row = evaluate_case(case_run)

    statuses = {d["metric"]: d["status"] for d in row["dimensions"] if d.get("metric")}
    assert statuses == {metric: "error" if metric in metrics else "passed" for metric in METRICS_IN_ORDER}
    assert all(d["error"] == "RuntimeError: Judge offline" for d in row["dimensions"] if d.get("metric") in metrics)
    assert all(d["name"] == d["metric"] for d in row["dimensions"] if d.get("metric"))
    assert service not in row["evidence_sha256"]
    assert not (case_run.directory / f"{service}.json").exists()


@title("Evidence bound to another sample is saved but reported as an error of its own metric")
def test_evaluate_case_isolates_unbound_evidence(case_run):
    case_run.reports["correctness"].return_value = {**case_run.evidence["correctness"], "sample_sha256": "wrong"}

    row = evaluate_case(case_run)

    statuses = {d["metric"]: d["status"] for d in row["dimensions"] if d.get("metric")}
    assert statuses == {
            "faithfulness": "passed",
            "factual_correctness": "error",
            "context_precision": "passed",
            "context_recall": "passed"}
    assert "correctness" in row["evidence_sha256"]


@title("A missing-information case runs the reviewed checks and marks every semantic metric not applicable")
def test_evaluate_case_skips_judges_for_refusal(case_run):
    gym = CASES["gym_missing"]
    save(case_run.sample_path, make_benchmark_sample(gym))

    row = evaluate_case(case_run, case=gym)

    assert [d["status"] for d in row["dimensions"]] == ["passed", "passed", *["not_applicable"] * 4]
    assert [(d["name"], d["metric"]) for d in row["dimensions"][2:]] == [(m, m) for m in sorted(MINIMA)]
    assert all("Abstention is checked against reviewed rules" in d["reason"] for d in row["dimensions"][2:])
    assert (row["judge_calls"], row["evidence_sha256"]) == (0, {})
    for report in case_run.reports.values():
        report.assert_not_called()


@title("A sample that fails its provenance check stops the case before any judge call")
def test_evaluate_case_rejects_unbound_sample(case_run):
    with pytest.raises(ValueError, match="model/digest mismatch"):
        evaluate_case(case_run, model="qwen2.5:7b")

    for report in case_run.reports.values():
        report.assert_not_called()
