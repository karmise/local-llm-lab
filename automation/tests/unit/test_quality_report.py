"""Saved-answer quality report: independent fact and source checks plus checksum-bound judge measurements."""

import hashlib
import json
import sys
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import Mock

import pytest

from llm_testkit import assertions
from llm_testkit.reporting import quality
from llm_testkit.reporting.allure_report import present_quality_report
from llm_testkit.reporting.quality import build_quality_report
from llm_testkit.reporting.steps import title
from test_support.builders.quality_report import (
        DOCUMENT, FULL_ANSWER, GENERATION_MODEL, JUDGE_DIGEST, QUESTION, policy_context, quality_inputs)
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit

FACTS, SOURCES, FAITHFULNESS = "Required answer facts", "Document sources", "Faithfulness measurement (no quality threshold)"
CORRECTNESS = "Factual correctness measurement (no quality threshold)"


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statuses(report: dict) -> list[str]:
    return [dimension["status"] for dimension in report["dimensions"]]


@title("A complete, cited and faithful answer passes the checks and records faithfulness without a threshold")
def test_report_passes_checks_and_records_measurement(tmp_path):
    inputs = quality_inputs(tmp_path)
    sample = json.loads(inputs.sample.read_text())

    report = build_quality_report(*inputs.paths())

    assert (report["schema_version"], report["status"], report["scenario"]) == (1, "checks_passed", "leave")
    assert report["dimensions"] == [{
            "name": FACTS,
            "status": "passed"}, {
            "name": SOURCES,
            "status": "passed",
            "details": {
            "expected_document_title": DOCUMENT}}, {
            "name": FAITHFULNESS,
            "status": "measured",
            "metric": "faithfulness",
            "details": {
            "value": 1.0,
            "statements": [FULL_ANSWER],
            "verdicts": [{
            "statement": FULL_ANSWER,
            "verdict": 1}],
            "judge_model": "test-judge",
            "judge_model_digest": JUDGE_DIGEST,
            "judge_configuration": {
            "think": False},
            "ragas_version": "test-version",
            "evaluated_at": "test-time",
            "threshold": None}}]
    assert (report["sample_sha256"], report["profile_sha256"]) == (sha256(inputs.sample), sha256(inputs.profile))
    assert report["evidence_sha256"] == {"faithfulness": sha256(inputs.evidence)}
    assert report["faithfulness_report_path"] == str(inputs.evidence.resolve())
    assert (report["correctness_report_path"], report["relevance_report_path"]) == (None, None)
    assert report["golden_dataset_sha256"] is None
    assert (report["generation_model"], report["question"],
            report["answer"]) == (GENERATION_MODEL, QUESTION, FULL_ANSWER)
    assert (report["contexts"], report["sources"]) == (sample["retrieved_contexts"], sample["response_sources"])
    assert report["metadata"] == {}
    assert report["interpretation"].startswith("Required facts and source checks; judge metrics are recorded")
    assert datetime.fromisoformat(report["created_at"]).utcoffset().total_seconds() == 0


@title("Perfect faithfulness does not hide a missing required fact")
def test_missing_fact_fails_report_despite_perfect_faithfulness(tmp_path):
    report = build_quality_report(*quality_inputs(tmp_path, answer="23 working days").paths())

    assert (report["status"], statuses(report)) == ("failed", ["failed", "passed", "measured"])
    assert "notice period" in report["dimensions"][0]["error"]
    with pytest.raises(AssertionError, match="notice period"):
        assertions.assert_quality_report(report)


@title("An error in one dimension outranks a failure in another")
def test_error_outranks_failure(tmp_path):
    inputs = quality_inputs(tmp_path, answer="23 working days")
    inputs.edit(inputs.evidence, status="error")

    report = build_quality_report(*inputs.paths())

    assert (report["status"], statuses(report)) == ("error", ["failed", "passed", "error"])


@pytest.mark.parametrize(
        "profile",
        [pytest.param({"schema_version": 2}, id="schema"),
        pytest.param({"question": "Other?"}, id="other-question")])
@title("A profile written for another question or schema is rejected [{param_id}]")
def test_report_rejects_mismatched_profile(tmp_path, profile):
    inputs = quality_inputs(tmp_path)
    inputs.edit(inputs.profile, **profile)

    with pytest.raises(ValueError, match="Quality profile does not match the captured question"):
        build_quality_report(*inputs.paths())


@pytest.mark.parametrize(("evidence", "error"), [
        pytest.param({"schema_version": 2}, "ValueError: Unsupported faithfulness report", id="schema-two"),
        pytest.param({"schema_version": True}, "ValueError: Unsupported faithfulness report", id="schema-boolean"),
        pytest.param({"metric": "answer_relevancy"}, "ValueError: Unsupported faithfulness report", id="other-metric"),
        pytest.param({"sample_sha256": "0" * 64}, "belongs to a different sample", id="other-sample"),
        pytest.param({"status": "error"}, "Faithfulness evaluation did not complete", id="unfinished"),
        pytest.param({"judge_calls": {}}, "differs from raw judge calls", id="calls-not-list"),
        pytest.param({"judge_calls": []}, "differs from raw judge calls", id="calls-missing")])
@title("Invalid faithfulness evidence is an independent error, not a failed answer [{param_id}]")
def test_invalid_faithfulness_evidence_is_an_error(tmp_path, evidence, error):
    inputs = quality_inputs(tmp_path)
    inputs.edit(inputs.evidence, **evidence)

    report = build_quality_report(*inputs.paths())

    assert (report["status"], statuses(report)) == ("error", ["passed", "passed", "error"])
    assert error in report["dimensions"][2]["error"]


@pytest.mark.parametrize(("result", "error"), [
        pytest.param({"value": 0.5}, "Faithfulness score does not match its verdicts", id="changed-score"),
        pytest.param({"verdicts": []}, "cover every extracted statement", id="missing-verdict")])
@title("A faithfulness result that disagrees with its own verdicts is an error [{param_id}]")
def test_inconsistent_faithfulness_result_is_an_error(tmp_path, result, error):
    inputs = quality_inputs(tmp_path)
    inputs.edit_result(**result)

    report = build_quality_report(*inputs.paths())

    assert report["dimensions"][2]["status"] == "error"
    assert error in report["dimensions"][2]["error"]


def judge_calls(statements, verdicts) -> list[dict]:
    return [{"output": {"statements": statements}}, {"output": {"statements": verdicts}}]


@pytest.mark.parametrize(
        "calls", [
        pytest.param(
        judge_calls(["different claim"], [{
        "statement": FULL_ANSWER,
        "verdict": 1}]), id="other-statements"),
        pytest.param(judge_calls([FULL_ANSWER], [{
        "statement": "different claim",
        "verdict": 0}]), id="other-verdicts"),
        pytest.param(judge_calls([FULL_ANSWER], [{
        "statement": FULL_ANSWER,
        "verdict": 1}]) * 2, id="extra-calls")])
@title("A saved summary that differs from the raw judge calls is an error [{param_id}]")
def test_summary_must_match_raw_judge_calls(tmp_path, calls):
    inputs = quality_inputs(tmp_path)
    inputs.edit(inputs.evidence, judge_calls=calls)

    report = build_quality_report(*inputs.paths())

    assert "ValueError: Faithfulness summary differs from raw judge calls" == report["dimensions"][2]["error"]


@title("Raw judge calls that match the saved summary are accepted")
def test_summary_matching_raw_judge_calls_is_measured(tmp_path):
    inputs = quality_inputs(tmp_path)
    inputs.edit(inputs.evidence, judge_calls=judge_calls([FULL_ANSWER], [{"statement": FULL_ANSWER, "verdict": 1}]))

    report = build_quality_report(*inputs.paths())

    assert report["dimensions"][2]["status"] == "measured"


@title("Missing faithfulness evidence is an error that names the missing file")
def test_missing_faithfulness_evidence_is_an_error(tmp_path):
    inputs = quality_inputs(tmp_path)
    inputs.evidence.unlink()

    report = build_quality_report(*inputs.paths())

    assert report["dimensions"][2]["error"].startswith("FileNotFoundError: ")
    assert report["evidence_sha256"] == {}


@title("The expected document comes from captured context, so a citation of another document fails")
def test_expected_source_is_derived_from_context_not_citations(tmp_path):
    inputs = quality_inputs(tmp_path, cited_document=f"automation-{'b' * 32}-company-policy.txt")

    report = build_quality_report(*inputs.paths())

    assert statuses(report) == ["passed", "failed", "measured"]


@pytest.mark.parametrize(
        "contexts", [
        pytest.param((FULL_ANSWER, ), id="no-metadata"),
        pytest.param((f"intro\n{policy_context()}", ), id="metadata-not-first"),
        pytest.param(("<document_metadata>\ntitle: policy\n</document_metadata>\nText", ), id="no-source-document"),
        pytest.param((policy_context("handbook.txt"), ), id="unexpected-document"),
        pytest.param(
        (policy_context(), policy_context(f"automation-{'b' * 32}-company-policy.txt")), id="two-documents")])
@title("Captured context must name exactly one expected policy document [{param_id}]")
def test_source_check_requires_one_expected_document(tmp_path, contexts):
    report = build_quality_report(*quality_inputs(tmp_path, contexts=contexts).paths())

    assert report["dimensions"][1] == {
            "name": SOURCES,
            "status": "error",
            "error": "ValueError: Expected exactly one policy document in captured context metadata"}


@title("The same expected document in several contexts counts once")
def test_source_check_accepts_repeated_document(tmp_path):
    report = build_quality_report(*quality_inputs(tmp_path, contexts=(policy_context(), policy_context())).paths())

    assert report["dimensions"][1]["status"] == "passed"


@pytest.fixture
def stub_evidence_checks(monkeypatch):
    """Replace dataset loading and evidence checks in the report module, recording how they are called."""
    calls = Mock()
    calls.load_golden_dataset.return_value = Mock(sha256="d" * 64)
    calls.check_correctness_evidence.return_value = {"value": 0.25}
    calls.check_relevance_evidence.return_value = {"context_precision": 0.5, "context_recall": 0.75}
    for name in ("load_golden_dataset", "check_correctness_evidence", "check_relevance_evidence"):
        monkeypatch.setattr(quality, name, getattr(calls, name))
    return calls


def write_evidence(path, data) -> object:
    path.write_text(json.dumps(data))
    return path


@title("Correctness evidence adds a measured dimension checked against the captured sample and golden dataset")
def test_correctness_evidence_adds_measured_dimension(tmp_path, stub_evidence_checks):
    inputs = quality_inputs(tmp_path)
    correctness = write_evidence(tmp_path / "correctness.json", {"observation": "correctness"})

    report = build_quality_report(*inputs.paths(), correctness_path=correctness)

    assert report["dimensions"][3] == {
            "name": CORRECTNESS,
            "status": "measured",
            "metric": "factual_correctness",
            "details": {
            "value": 0.25}}
    evidence, checksum, sample, dataset = stub_evidence_checks.check_correctness_evidence.call_args.args
    assert (evidence, checksum) == ({"observation": "correctness"}, sha256(inputs.sample))
    assert (sample["response"], dataset) == (FULL_ANSWER, stub_evidence_checks.load_golden_dataset.return_value)
    assert report["evidence_sha256"]["correctness"] == sha256(correctness)
    assert report["correctness_report_path"] == str(correctness.resolve())
    assert report["golden_dataset_sha256"] == "d" * 64


@title("By default the golden dataset and policy are read next to the profile")
def test_golden_dataset_defaults_to_profile_directory(tmp_path, stub_evidence_checks):
    inputs = quality_inputs(tmp_path)

    build_quality_report(*inputs.paths(), correctness_path=write_evidence(tmp_path / "correctness.json", {}))

    stub_evidence_checks.load_golden_dataset.assert_called_once_with(
            tmp_path / "golden-policy.json", tmp_path / "company-policy.txt")


@title("An explicit golden dataset and policy override the profile directory")
def test_golden_dataset_paths_can_be_given(tmp_path, stub_evidence_checks):
    inputs = quality_inputs(tmp_path)

    build_quality_report(
            *inputs.paths(), correctness_path=write_evidence(tmp_path / "correctness.json", {}),
            golden_dataset_path=tmp_path / "dataset.json", policy_file=tmp_path / "policy.txt")

    stub_evidence_checks.load_golden_dataset.assert_called_once_with(tmp_path / "dataset.json", tmp_path / "policy.txt")


@title("Relevance evidence adds context precision and recall from one validated read")
def test_relevance_dimensions_share_one_validated_evidence(tmp_path, stub_evidence_checks):
    inputs = quality_inputs(tmp_path)
    relevance = write_evidence(tmp_path / "relevance.json", {"observation": "relevance"})
    checked = stub_evidence_checks.check_relevance_evidence.return_value

    report = build_quality_report(*inputs.paths(), relevance_path=relevance)

    assert report["dimensions"][3:] == [{
            "name": metric,
            "status": "measured",
            "metric": metric,
            "details": {
            "value": value,
            "threshold": None,
            "metric": metric,
            "evidence": checked}} for metric, value in (("context_precision", 0.5), ("context_recall", 0.75))]
    stub_evidence_checks.check_relevance_evidence.assert_called_once()
    assert stub_evidence_checks.check_relevance_evidence.call_args.args[:2] == ({
            "observation": "relevance"}, sha256(inputs.sample))
    assert report["evidence_sha256"]["relevance"] == sha256(relevance)
    assert report["relevance_report_path"] == str(relevance.resolve())


@title("Correctness and relevance evidence share one golden dataset load")
def test_correctness_and_relevance_share_dataset(tmp_path, stub_evidence_checks):
    inputs = quality_inputs(tmp_path)

    report = build_quality_report(
            *inputs.paths(), correctness_path=write_evidence(tmp_path / "correctness.json", {}),
            relevance_path=write_evidence(tmp_path / "relevance.json", {}))

    assert [dimension.get("metric") for dimension in report["dimensions"]
            ] == [None, None, "faithfulness", "factual_correctness", "context_precision", "context_recall"]
    stub_evidence_checks.load_golden_dataset.assert_called_once()


@title("Missing relevance evidence creates two visible errors without hiding the other checks")
def test_missing_relevance_evidence_is_two_errors(tmp_path):
    report = build_quality_report(*quality_inputs(tmp_path).paths(), relevance_path=tmp_path / "absent.json")

    assert statuses(report) == ["passed", "passed", "measured", "error", "error"]
    assert [d["name"] for d in report["dimensions"][-2:]] == ["context_precision", "context_recall"]


@title("Quality gates reject a report whose gated semantic evidence was not supplied")
def test_saved_report_gates_require_all_evidence(tmp_path):
    gates = AUTOMATION_ROOT / "test_data/quality-gates.json"

    report = build_quality_report(*quality_inputs(tmp_path).paths(), gates_path=gates)

    assert (report["status"], report["dimensions"][2]["status"], len(report["dimensions"])) == ("error", "passed", 6)
    with pytest.raises(AssertionError, match="not supplied"):
        assertions.assert_quality_report(report)


@title("Allure renders every quality dimension before failing on the one that failed")
def test_allure_renders_remaining_steps_after_a_failed_dimension(tmp_path, monkeypatch):
    report = build_quality_report(*quality_inputs(tmp_path, answer="23 working days").paths())
    visited = []

    @contextmanager
    def step(name):
        visited.append(name)
        yield

    monkeypatch.setitem(sys.modules, "allure", Mock(step=step))

    with pytest.raises(AssertionError, match="notice period"):
        present_quality_report(report)
    assert visited == [FACTS, SOURCES, FAITHFULNESS]


@title("The source document is found in multi-line context metadata")
def test_source_check_reads_multiline_metadata(tmp_path):
    context = f"<document_metadata>\ntitle: Policy\nsourceDocument: {DOCUMENT}\n</document_metadata>\n{FULL_ANSWER}"

    report = build_quality_report(*quality_inputs(tmp_path, contexts=(context, )).paths())

    assert report["dimensions"][1]["details"] == {"expected_document_title": DOCUMENT}


@title("Run metadata saved with the sample is carried into the report")
def test_report_carries_run_metadata(tmp_path):
    inputs = quality_inputs(tmp_path)
    inputs.edit(inputs.sample, metadata={"run": "nightly"})
    inputs.rebind_evidence()

    report = build_quality_report(*inputs.paths())

    assert (report["metadata"], report["status"]) == ({"run": "nightly"}, "checks_passed")
