"""Benchmark summary: row outcomes, the planned denominator, judge controls, the Markdown report and review sheet."""

from unittest.mock import Mock, call

import pytest

from llm_testkit.reporting import benchmark as reporting
from llm_testkit.reporting.benchmark import markdown, review_worksheet, row_status, summarize
from llm_testkit.reporting.steps import title
from test_support.builders.benchmark import MINIMA, MODEL, make_calibration, make_definition, make_row

pytestmark = pytest.mark.unit

METRICS = ["context_precision", "context_recall", "factual_correctness", "faithfulness"]


@pytest.fixture
def definition() -> dict:
    return make_definition()


@pytest.fixture
def calibration(definition) -> dict:
    return make_calibration(definition)


def complete_rows() -> list[dict]:
    return [make_row(), make_row("gym_missing", "missing_information")]


def metric(row: dict, name: str) -> dict:
    return next(d for d in row["dimensions"] if d.get("metric") == name)


@pytest.mark.parametrize(("change", "status"), [
        pytest.param({}, "passed", id="all-passed"),
        pytest.param({"error": "Generation crashed"}, "error", id="row-error"),
        pytest.param({"generation_status": "error"}, "error", id="generation-error"),
        pytest.param({"generation_status": "skipped"}, "error", id="generation-skipped"),
        pytest.param({"generation_status": "failed"}, "failed", id="generation-failed"),
        pytest.param({"dimensions": []}, "error", id="no-dimensions")])
@title("A row is an error, a failure or a pass from its generation and dimension outcomes [{param_id}]")
def test_row_status(change, status):
    assert row_status({**make_row(), **change}) == status


@title("A row that crashed before producing dimensions is an error")
def test_row_status_without_dimensions():
    row = make_row()
    del row["dimensions"]

    assert row_status(row) == "error"


@pytest.mark.parametrize(("dimension_status", "status"), [("failed", "failed"), ("error", "error"),
        ("not_applicable", "passed")])
@title("One failed or errored dimension decides the row outcome; not-applicable does not [{param_id}]")
def test_row_status_from_one_dimension(dimension_status, status):
    row = make_row()
    row["dimensions"][2]["status"] = dimension_status

    assert row_status(row) == status


@title("A complete passing matrix passes, with every metric measured once and refusals not applicable")
def test_summary_of_complete_matrix(definition, calibration):
    rows = complete_rows()

    report = summarize(definition, rows, calibration)

    assert (report["schema_version"], report["status"], report["judge_controls_matched"]) == (1, "checks_passed", True)
    assert (report["manifest"], report["calibration"], report["results"]) == (definition, calibration, rows)
    assert report["missing_rows"] == []
    assert "No population accuracy or clinical validity claim" in report["interpretation"]
    summary = report["summary"]
    assert {
            key: summary[key]
            for key in ("planned", "passed", "failed", "errors", "missing", "pass_rate")} == {
            "planned": 2,
            "passed": 2,
            "failed": 0,
            "errors": 0,
            "missing": 0,
            "pass_rate": 1.0}
    assert summary["metrics"]["faithfulness"] == {
            "eligible": 1,
            "measured": 1,
            "unavailable": 0,
            "not_applicable": 1,
            "below_minimum": 0,
            "mean": 1.0,
            "minimum": 1.0}
    assert list(summary["metrics"]) == METRICS
    assert list(report["models"]) == [MODEL]
    assert list(report["categories"]) == ["multi_fact", "missing_information"]
    assert report["categories"]["missing_information"]["metrics"]["faithfulness"]["mean"] is None


@title("A missing result counts against the planned denominator and makes the benchmark an error")
def test_summary_keeps_missing_rows(definition, calibration):
    report = summarize(definition, [make_row()], calibration)

    assert report["status"] == "error"
    assert (report["summary"]["missing"], report["summary"]["pass_rate"]) == (1, 0.5)
    assert report["missing_rows"] == [definition["expected_rows"][1]]


@title("A metric below its threshold fails the benchmark and is counted, while its mean stays visible")
def test_summary_counts_metric_below_threshold(definition, calibration):
    rows = complete_rows()
    metric(rows[0], "faithfulness").update(value=0.5, status="failed")

    report = summarize(definition, rows, calibration)

    faithfulness = report["summary"]["metrics"]["faithfulness"]
    assert report["status"] == "failed"
    assert (faithfulness["below_minimum"], faithfulness["mean"], faithfulness["minimum"]) == (1, 0.5, 0.5)


@title("A failed answer check cannot be hidden by perfect metric means")
def test_summary_keeps_failed_check_visible(definition, calibration):
    rows = complete_rows()
    rows[0]["dimensions"][0].update(status="failed", error="Missing annual allowance")

    report = summarize(definition, rows, calibration)

    assert report["status"] == "failed"
    assert report["summary"]["failed"] == 1
    assert report["summary"]["metrics"]["faithfulness"]["mean"] == 1.0


@title("A metric error makes the benchmark an error and leaves that metric unavailable")
def test_summary_reports_metric_error_as_unavailable(definition, calibration):
    rows = complete_rows()
    rows[0]["dimensions"][2] = {"name": "context_precision", "metric": "context_precision", "status": "error"}

    report = summarize(definition, rows, calibration)

    assert report["status"] == "error"
    assert report["summary"]["metrics"]["context_precision"]["unavailable"] == 1
    assert report["summary"]["metrics"]["context_precision"]["mean"] is None


@title("A row that failed before evaluation is kept as an error without validating its dimensions")
def test_summary_keeps_errored_row(definition, calibration):
    crashed = {key: value for key, value in make_row().items() if key != "dimensions"}
    rows = [{**crashed, "error": "Generation crashed"}, complete_rows()[1]]

    report = summarize(definition, rows, calibration)

    assert (report["status"], report["summary"]["errors"]) == ("error", 1)
    assert report["summary"]["metrics"]["faithfulness"]["unavailable"] == 1


@title("Rows after an errored row are still validated")
def test_summary_validates_rows_after_errored_row(definition, calibration):
    refusal = complete_rows()[1]
    refusal["dimensions"][2]["status"] = "passed"  # a refusal's semantic metrics must be not applicable

    with pytest.raises(ValueError, match="Only refusal metrics"):
        summarize(definition, [{**make_row(), "error": "Generation crashed"}, refusal], calibration)


@title("A missing answered case leaves its metrics unavailable rather than failing the summary")
def test_summary_without_answered_case(definition, calibration):
    report = summarize(definition, [complete_rows()[1]], calibration)

    assert report["summary"]["metrics"]["faithfulness"]["unavailable"] == 1
    assert report["summary"]["metrics"]["faithfulness"]["mean"] is None


@title("A score exactly equal to its threshold is a pass")
def test_summary_accepts_score_on_threshold(definition, calibration):
    rows = complete_rows()
    metric(rows[0], "faithfulness")["value"] = MINIMA["faithfulness"]

    assert summarize(definition, rows, calibration)["status"] == "checks_passed"


@title("Sub-second answer timings are valid measurements")
def test_summary_accepts_sub_second_timing(definition, calibration):
    rows = complete_rows()
    rows[0]["answer_request_seconds"] = 0.5

    assert summarize(definition, rows, calibration)["summary"]["answer_timing"]["mean_seconds"] == 0.5


@pytest.mark.parametrize(
        "change", [
        pytest.param({"status": "mismatch"}, id="mismatch"),
        pytest.param({"judge_model": "other"}, id="other-judge"),
        pytest.param({"judge_model_digest": ""}, id="no-judge-digest"),
        pytest.param({"controls_sha256": "other"}, id="other-catalog"),
        pytest.param({"control_ids": ["wrong_numbers"]}, id="other-controls"),
        pytest.param({"results": [{
        "status": "matched"}]}, id="missing-control-results"),
        pytest.param({"results": [{
        "status": "matched"}, {
        "status": "matched"}, {
        "status": "error"}]}, id="failed-control")])
@title("A benchmark cannot pass unless its judge controls matched for the same judge and catalog [{param_id}]")
def test_summary_requires_matched_judge_controls(definition, calibration, change):
    report = summarize(definition, complete_rows(), {**calibration, **change})

    assert (report["status"], report["judge_controls_matched"]) == ("error", False)


@title("Answer timings are summarised as count, mean, minimum and maximum; missing timings stay unavailable")
def test_summary_answer_timing(definition, calibration):
    rows = complete_rows()
    rows[0]["answer_request_seconds"], rows[1]["answer_request_seconds"] = 10.0, 20.0

    timing = summarize(definition, rows, calibration)["summary"]["answer_timing"]
    legacy = summarize(definition, complete_rows(), calibration)["summary"]["answer_timing"]

    assert {
            key: timing[key]
            for key in ("measured", "unavailable", "mean_seconds", "minimum_seconds", "maximum_seconds")} == {
            "measured": 2,
            "unavailable": 0,
            "mean_seconds": 15.0,
            "minimum_seconds": 10.0,
            "maximum_seconds": 20.0}
    assert "excludes fixture setup, judge calls and cleanup" in timing["scope"]
    assert (legacy["measured"], legacy["unavailable"], legacy["mean_seconds"]) == (0, 2, None)
    assert (legacy["minimum_seconds"], legacy["maximum_seconds"]) == (None, None)


@pytest.mark.parametrize("duration", [0, -1, float("nan"), True], ids=["zero", "negative", "nan", "boolean"])
@title("An invalid recorded answer duration is rejected [{param_id}]")
def test_summary_rejects_invalid_duration(definition, calibration, duration):
    rows = complete_rows()
    rows[0]["answer_request_seconds"] = duration

    with pytest.raises(ValueError, match="finite positive"):
        summarize(definition, rows, calibration)


def first_metric(row: dict) -> dict:
    return row["dimensions"][2]


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(lambda rows: rows.append(make_row()), "Unknown or duplicate", id="duplicate-row"),
        pytest.param(lambda rows: rows[0].update(case_id="unknown"), "Unknown or duplicate", id="unplanned-row"),
        pytest.param(lambda rows: rows[0].update(model="other"), "Unknown or duplicate", id="unplanned-model"),
        pytest.param(lambda rows: rows[0].update(category="boundary"), "category mismatch", id="category"),
        pytest.param(lambda rows: rows[0]["dimensions"].pop(), "two checks and four metric", id="missing-metric"),
        pytest.param(
        lambda rows: rows[0]["dimensions"].append({
        "name": "Extra",
        "status": "passed"}), "two checks and four metric", id="extra-check"),
        pytest.param(
        lambda rows: first_metric(rows[0]).update(metric="faithfulness"), "two checks and four metric",
        id="repeated-metric"),
        pytest.param(
        lambda rows: rows[0]["dimensions"][0].update(status="skipped"), "Unknown benchmark dimension",
        id="unknown-status"),
        pytest.param(
        lambda rows: first_metric(rows[0]).update(status="not_applicable"), "Only refusal metrics",
        id="answered-metric-not-applicable"),
        pytest.param(
        lambda rows: first_metric(rows[1]).update(status="passed"), "Only refusal metrics",
        id="refusal-metric-measured"),
        pytest.param(lambda rows: first_metric(rows[0]).update(minimum=0.2), "threshold differs", id="threshold"),
        pytest.param(
        lambda rows: first_metric(rows[0]).update(value=0.1), "status differs from its metric score", id="false-pass"),
        pytest.param(
        lambda rows: first_metric(rows[0]).update(value=0.9, status="failed"), "status differs from its metric score",
        id="false-failure"),
        pytest.param(
        lambda rows: rows[0]["dimensions"][1].update(status="not_applicable"), "always required",
        id="source-check-skipped")])
@title("Inconsistent identities, dimensions, thresholds or statuses are rejected [{param_id}]")
def test_summary_rejects_inconsistent_rows(definition, calibration, corrupt, message):
    rows = complete_rows()
    corrupt(rows)

    with pytest.raises(ValueError, match=message):
        summarize(definition, rows, calibration)


@title("A score that is not a finite number from 0 to 1 is rejected")
def test_summary_rejects_invalid_score(definition, calibration):
    rows = complete_rows()
    first_metric(rows[0])["value"] = float("nan")

    with pytest.raises(AssertionError, match="finite quality score"):
        summarize(definition, rows, calibration)


@pytest.mark.parametrize(("change", "message"), [
        pytest.param({"expected_rows": []}, "unique, nonempty matrix", id="empty-matrix"),
        pytest.param({"expected_rows": [{
        "case_id": "paid_leave",
        "category": "multi_fact",
        "model": MODEL}] * 2}, "unique, nonempty matrix", id="duplicate-planned-row"),
        pytest.param({"quality_gates": {
        "minimum_scores": {
        "faithfulness": 0.9}}}, "all four metric thresholds", id="missing-thresholds")])
@title("A manifest without a unique matrix or all four thresholds is rejected [{param_id}]")
def test_summary_rejects_invalid_manifest(definition, calibration, change, message):
    with pytest.raises(ValueError, match=message):
        summarize({**definition, **change}, complete_rows(), calibration)


@title("Two generation models are compared only under the same workspace settings, with per-model outcomes")
def test_summary_compares_two_models(calibration):
    definition = make_definition(case_ids=("paid_leave", ), models=(MODEL, "qwen2.5:7b"))
    first, second = make_row(), make_row(model="qwen2.5:7b")
    second["model_digest"] = "other-weights"

    report = summarize(definition, [first, second], calibration)

    assert report["status"] == "checks_passed"
    assert list(report["models"]) == [MODEL, "qwen2.5:7b"]
    assert all(group["passed"] == 1 for group in report["models"].values())


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(
        lambda first, second: second["workspace_configuration"].update(openAiPrompt="Other prompt"),
        "different workspace settings", id="other-settings-for-other-model"),
        pytest.param(
        lambda first, second: second.update(
        model=MODEL, case_id="gym_missing", category="missing_information", workspace_configuration={
        "openAiPrompt": "Other"}), "changed within one model", id="settings-changed-within-model"),
        pytest.param(
        lambda first, second: second.update(
        model=MODEL, case_id="gym_missing", category="missing_information", model_digest="changed"),
        "digest changed within", id="weights-changed")])
@title("Changed prompts or generation weights invalidate a comparison [{param_id}]")
def test_summary_rejects_changed_generation_setup(corrupt, message):
    definition = make_definition(case_ids=("paid_leave", "gym_missing"), models=(MODEL, "qwen2.5:7b"))
    first, second = make_row(), make_row("gym_missing", "missing_information", model="qwen2.5:7b")
    corrupt(first, second)

    with pytest.raises(ValueError, match=message):
        summarize(definition, [first, second], make_calibration(definition))


@pytest.fixture
def report(definition, calibration) -> dict:
    rows = complete_rows()
    rows[0]["answer_request_seconds"] = 10.0
    metric(rows[0], "context_recall").update(status="error", error="Judge offline")
    metric(rows[0], "context_recall").pop("value")
    metric(rows[1], "faithfulness")["reason"] = "Refusal scope"
    summarised = summarize(definition, rows, calibration)
    # Rendering also has to cope with a row that failed before producing any dimension.
    crashed = {"case_id": "unplanned", "model": MODEL, "error": "Generation crashed"}
    return {**summarised, "results": [*rows, crashed]}


@title("The Markdown report states the status and tabulates models, timing, metrics, categories and judge controls")
def test_markdown_tables(report):
    lines = markdown(report).splitlines()

    assert lines[:5] == ["# Policy quality benchmark", "", "Status: **error**", "", report["interpretation"]]
    assert "| qwen3.5:4b | 2 | 1 | 0 | 1 | 0 | 50.0% |" in lines
    assert "| qwen3.5:4b | 1 / 2 | 1 | 10.000 | 10.000 | 10.000 |" in lines
    assert "| qwen3.5:4b | faithfulness | 1.000 | 1.000 | 1 / 1 | 0 | 0 | 1 |" in lines
    assert "| qwen3.5:4b | context_recall | N/A | N/A | 0 / 1 | 1 | 0 | 1 |" in lines
    assert "| multi_fact | 1 | 0 | 0 | 1 | 0 |" in lines
    assert "| missing_information | 1 | 1 | 0 | 0 | 0 |" in lines
    assert any(line.startswith("Status: matched. This is a small hand-labelled") for line in lines)


@title("Case diagnostics show each row's status, timing, error and every dimension with its detail")
def test_markdown_case_diagnostics(report):
    lines = markdown(report).splitlines()

    assert "### paid_leave / qwen3.5:4b" in lines
    assert "Answer request: 10.000 seconds." in lines
    assert "- faithfulness: passed. 1.0" in lines
    assert "- context_recall: error. Judge offline" in lines
    assert "- Reviewed answer rules: passed. " in lines
    assert "- faithfulness: not_applicable. Refusal scope" in lines
    assert ["### unplanned / qwen3.5:4b", "", "Status: error", "", "Generation crashed"
            ] == lines[lines.index("### unplanned / qwen3.5:4b"):lines.index("### unplanned / qwen3.5:4b") + 5]
    assert markdown(report).endswith("\n")


@title("The Markdown report has its section headings and table headers in order")
def test_markdown_sections(report):
    lines = markdown(report).splitlines()
    expected = [
            "## Results by generation model", "| Model | Planned | Passed | Failed | Errors | Missing | Pass rate |",
            "## Answer request timing",
            "| Model | Measured / planned | Unavailable | Mean seconds | Min seconds | Max seconds |",
            "## Semantic metrics by model",
            "Means cover available measurements only. Check unavailable counts before comparing.",
            "| Model | Metric | Mean | Minimum | Measured / eligible | Unavailable | Below threshold | N/A |",
            "## Results by question category", "| Category | Planned | Passed | Failed | Errors | Missing |",
            "## Judge control check", "## Case diagnostics"]

    positions = [lines.index(line) for line in expected]

    assert positions == sorted(positions)
    assert any(
            line.startswith("Descriptive timings include retrieval, model loading and generation.") for line in lines)


@title("Missing timings and an unavailable judge-control status are rendered as N/A and unavailable")
def test_markdown_without_timing_or_calibration(definition, calibration):
    report = summarize(definition, complete_rows(), calibration)
    report["calibration"] = {}

    lines = markdown(report).splitlines()

    assert "| qwen3.5:4b | 0 / 2 | 2 | N/A | N/A | N/A |" in lines
    assert any(line.startswith("Status: unavailable. This is a small") for line in lines)


@title("The review worksheet lists every result for a pending human verdict and flags judge disagreements")
def test_review_worksheet(report):
    worksheet = review_worksheet(report)
    disagreement, refusal, crashed = worksheet["rows"]

    assert (worksheet["schema_version"], worksheet["status"]) == (1, "pending_human_review")
    assert "Keep this run's evidence immutable" in worksheet["instruction"]
    assert (disagreement["case_id"], disagreement["model"]) == ("paid_leave", MODEL)
    assert disagreement["review_flags"] == ["semantic_disagreement_candidate"]
    assert (refusal["review_flags"], crashed["review_flags"]) == ([], [])
    for row in worksheet["rows"]:
        assert (row["human_verdict"], row["reviewer"], row["notes"],
                row["judge_disagreements"]) == (None, None, None, [])
        assert set(row) == {
                "case_id", "model", "question", "reference", "answer", "human_verdict", "review_flags",
                "judge_disagreements", "reviewer", "notes"}


def fail_first_metric(row: dict) -> None:
    row["dimensions"][2].update(status="failed", value=0.5)


def crash_after_metric_failure(row: dict) -> None:
    fail_first_metric(row)
    row["error"] = "Crashed"


@pytest.mark.parametrize(("change", "flags"), [
        pytest.param(fail_first_metric, ["semantic_disagreement_candidate"], id="first-metric-failed"),
        pytest.param(crash_after_metric_failure, [], id="errored-row"),
        pytest.param(lambda row: row.pop("dimensions"), [], id="no-dimensions")])
@title("Only rows whose two checks passed but a metric failed are flagged for judge review [{param_id}]")
def test_review_worksheet_flags(change, flags):
    row = make_row()
    change(row)

    assert review_worksheet({"results": [row]})["rows"][0]["review_flags"] == flags


@title("A metric failure after a failed answer check is not flagged as a judge disagreement")
def test_review_worksheet_requires_passed_checks_for_flag(definition, calibration):
    row = make_row()
    row["dimensions"][0]["status"] = "failed"
    metric(row, "faithfulness").update(status="failed", value=0.5)
    row.update(question="Q", reference="R", answer="A")

    worksheet = review_worksheet({"results": [row]})

    assert worksheet["rows"][0]["review_flags"] == []
    assert (worksheet["rows"][0]["question"], worksheet["rows"][0]["reference"],
            worksheet["rows"][0]["answer"]) == ("Q", "R", "A")


@title("Presenting a passing benchmark attaches the results, the summary and every case")
def test_present_passing_benchmark(definition, calibration, monkeypatch):
    attach, metadata = Mock(), Mock()
    monkeypatch.setattr(reporting, "attach_text", attach)
    monkeypatch.setattr(reporting, "set_metadata", metadata)
    report = summarize(definition, complete_rows(), calibration)

    reporting.present_benchmark_report(report)

    metadata.assert_called_once_with(feature="RAG quality", story="Dataset benchmark")
    names = [attached.kwargs["name"] for attached in attach.call_args_list]
    assert names == [
            "Benchmark results and case diagnostics", "Benchmark summary", "paid_leave / qwen3.5:4b",
            "gym_missing / qwen3.5:4b"]
    assert attach.call_args_list[1] == call(markdown(report), name="Benchmark summary")


@title("Presenting a failed benchmark attaches every case before the aggregate check fails")
def test_present_failed_benchmark(definition, calibration, monkeypatch):
    attach = Mock()
    monkeypatch.setattr(reporting, "attach_text", attach)
    rows = complete_rows()
    rows[0]["dimensions"][0].update(status="failed", error="Missing annual allowance")
    report = summarize(definition, rows, calibration)

    with pytest.raises(AssertionError, match="did not pass: failed"):
        reporting.present_benchmark_report(report)

    assert [attached.kwargs["name"]
            for attached in attach.call_args_list][-2:] == ["paid_leave / qwen3.5:4b", "gym_missing / qwen3.5:4b"]


@title("A report without a summary is attached without a Markdown summary")
def test_present_report_without_summary(monkeypatch):
    attach = Mock()
    monkeypatch.setattr(reporting, "attach_text", attach)

    with pytest.raises(AssertionError, match="did not pass: error"):
        reporting.present_benchmark_report({"schema_version": 1, "status": "error", "results": []})

    assert [attached.kwargs["name"] for attached in attach.call_args_list] == ["Benchmark results and case diagnostics"]


@title("Thresholds used by the summary come from the reviewed quality gates")
def test_row_thresholds_match_gates():
    assert {d["metric"]: d["minimum"] for d in make_row()["dimensions"] if d.get("metric")} == MINIMA
