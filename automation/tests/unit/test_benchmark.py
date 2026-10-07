import hashlib
import json
import shutil

import pytest

from llm_testkit import assertions
from llm_testkit.datasets.benchmark import make_plan, manifest
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.evaluation import benchmark as evaluation
from llm_testkit.evaluation import benchmark_runner as runner
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.benchmark import markdown, review_worksheet, summarize
from llm_testkit.reporting.benchmark_evidence import load_saved_benchmark
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import mocks as mock_checks
from test_support.assertions import values as value_checks
from test_support.assertions.benchmark import check_benchmark_configuration_changes
from test_support.builders.benchmark import (
    _mock_metrics,
    _row,
    _sample,
    make_calibrate_stub,
    make_forged_benchmark_summary,
    make_generate_stub,
    make_subprocess_run_stub,
    prepare_invalid_summary_case,
)
from test_support.data.benchmark import (
    INVALID_PLAN_OPTIONS_CASES,
    INVALID_SUMMARY_CHANGE_CASES,
    ROOT,
)
from test_support.fixtures.unit_benchmark import benchmark_data as benchmark_data

pytestmark = pytest.mark.unit


@title("Benchmark default matrix has a bounded serial model-call budget")
def test_default_plan_budget(benchmark_data):
    dataset = benchmark_data[0]
    value_checks.equal(make_plan(dataset).maximum_calls, 43)
    value_checks.equal(
        make_plan(dataset, models=["qwen3.5:4b", "qwen2.5:7b"], max_model_calls=80).maximum_calls,
        80,
    )


@pytest.mark.parametrize(
    "options",
    INVALID_PLAN_OPTIONS_CASES,
)
@title("Benchmark rejects invalid or over-budget matrices before model calls [{param_id}]")
def test_invalid_plan(benchmark_data, options):
    errors.rejects(lambda: make_plan(benchmark_data[0], **options), expected=ValueError)


@title("Benchmark summary excludes refusal metrics explicitly and uses the planned denominator")
def test_summary_and_missing_rows(benchmark_data):
    _, _, _, definition, calibration = benchmark_data
    report = summarize(definition, [_row()], calibration)
    value_checks.equal(report["status"], "error")
    value_checks.equal(report["summary"]["pass_rate"], 0.5)
    value_checks.equal(report["summary"]["missing"], 1)
    value_checks.equal(report["summary"]["metrics"]["context_recall"]["not_applicable"], 1)
    value_checks.contains(markdown(report), "1 / 1")
    complete = summarize(
        definition, [_row(), _row("gym_missing", "missing_information")], calibration
    )
    assertions.assert_benchmark_report(complete)
    value_checks.identical(
        complete["categories"]["missing_information"]["metrics"]["faithfulness"]["mean"], None
    )
    value_checks.equal(review_worksheet(complete)["status"], "pending_human_review")


@title("Failed answer checks and judge errors cannot be hidden by perfect mean scores")
def test_failures_remain_visible(benchmark_data):
    _, _, _, definition, calibration = benchmark_data
    row = _row()
    row["dimensions"][0].update(status="failed", error="Missing annual allowance")
    report = summarize(definition, [row, _row("gym_missing", "missing_information")], calibration)
    value_checks.equal(report["status"], "failed")
    value_checks.equal(report["summary"]["metrics"]["faithfulness"]["mean"], 1)
    row["dimensions"][2] = {
        "name": "context_precision",
        "metric": "context_precision",
        "status": "error",
    }
    report = summarize(definition, [row], calibration)
    value_checks.equal(report["summary"]["metrics"]["context_precision"]["unavailable"], 1)
    value_checks.identical(report["summary"]["metrics"]["context_precision"]["mean"], None)


@pytest.mark.parametrize(
    "change",
    INVALID_SUMMARY_CHANGE_CASES,
)
@title("Benchmark refuses inconsistent case identities and metric outcomes [{param_id}]")
def test_invalid_summary(benchmark_data, change):
    _, _, _, definition, calibration = benchmark_data
    row = _row()
    rows = [row]
    prepare_invalid_summary_case(change, row, rows)
    errors.rejects(
        lambda: summarize(definition, rows, calibration), expected=(ValueError, AssertionError)
    )


@title("Model comparisons reject changed prompts or generation weights")
def test_different_configurations_rejected(benchmark_data):
    _, _, _, definition, calibration = benchmark_data
    check_benchmark_configuration_changes(calibration, definition)


@title("Judge control mismatch prevents a benchmark from claiming acceptance")
def test_control_mismatch(benchmark_data):
    _, _, _, definition, calibration = benchmark_data
    calibration["results"][0]["status"] = "mismatch"
    report = summarize(
        definition, [_row(), _row("gym_missing", "missing_information")], calibration
    )
    value_checks.equal(report["status"], "error")
    value_checks.identical(report["judge_controls_matched"], False)


@title("All four benchmark metrics validate original evidence and preserve independent failures")
def test_case_pipeline(tmp_path, benchmark_data, monkeypatch, unit_settings):
    dataset, gates, _, _, _ = benchmark_data
    case = dataset.cases[0]
    sample_path = tmp_path / "sample.json"
    write_sample(sample_path, _sample(case, dataset))
    _mock_metrics(monkeypatch, case, dataset, sample_path)
    kwargs = dict(
        directory=tmp_path,
        dataset=dataset,
        dataset_path=ROOT / "test_data/golden-policy.json",
        policy_file=ROOT / "test_data/company-policy.txt",
        case=case,
        model="qwen3.5:4b",
        judge_model="qwen3.5:4b",
        judge_digest="judge-digest",
        settings=unit_settings,
        minima=gates["minimum_scores"],
    )
    row = evaluation.evaluate_case(sample_path, **kwargs)
    value_checks.length(row["dimensions"], 6)
    value_checks.all_true((d["status"] == "passed" for d in row["dimensions"]))
    value_checks.equal(row["judge_calls"], 8)
    value_checks.length(row["evidence_sha256"], 3)
    other = tmp_path / "failed"
    other.mkdir()
    evaluation.evaluate_correctness_report.return_value["sample_sha256"] = "wrong"
    row = evaluation.evaluate_case(sample_path, **{**kwargs, "directory": other})
    value_checks.equal(
        next((d for d in row["dimensions"] if d.get("metric") == "factual_correctness"))["status"],
        "error",
    )
    value_checks.equal(
        next((d for d in row["dimensions"] if d.get("metric") == "context_recall"))["status"],
        "passed",
    )


@title("Refusal cases run reviewed checks without invoking a semantic judge")
def test_refusal_case_no_judge(tmp_path, benchmark_data, monkeypatch, mock_factory, unit_settings):
    dataset, gates, _, _, _ = benchmark_data
    case = next(c for c in dataset.cases if c.id == "gym_missing")
    path = tmp_path / "sample.json"
    write_sample(path, _sample(case, dataset))
    judge = mock_factory(side_effect=AssertionError("Unexpected judge call"))
    monkeypatch.setattr(evaluation, "evaluate_sample_report", judge)
    row = evaluation.evaluate_case(
        path,
        directory=tmp_path,
        dataset=dataset,
        dataset_path=ROOT / "test_data/golden-policy.json",
        policy_file=ROOT / "test_data/company-policy.txt",
        case=case,
        model="qwen3.5:4b",
        judge_model="qwen3.5:4b",
        judge_digest="judge-digest",
        settings=unit_settings,
        minima=gates["minimum_scores"],
    )
    value_checks.equal(row["judge_calls"], 0)
    value_checks.equal(sum((d["status"] == "not_applicable" for d in row["dimensions"])), 4)
    mock_checks.not_called(judge)


@title("Benchmark runner preserves its full matrix, snapshots and human review worksheet")
def test_runner_pipeline(tmp_path, benchmark_data, monkeypatch, mock_factory):
    dataset, gates, plan, _, calibration = benchmark_data
    root = tmp_path / "automation"
    shutil.copytree(ROOT / "test_data", root / "test_data")
    catalog = mock_factory(status_code=200)
    catalog.json.return_value = {"models": [{"name": "qwen3.5:4b", "digest": "judge-digest"}]}
    monkeypatch.setattr(runner.OllamaClient, "list_models", lambda _: catalog)

    calibrate = make_calibrate_stub(calibration)

    monkeypatch.setattr(runner, "calibrate", calibrate)

    generate = make_generate_stub(dataset, monkeypatch)

    monkeypatch.setattr(runner, "generate_sample", generate)
    output = tmp_path / "run"
    report = runner.run(root, output, plan, dataset, gates, notify=lambda *a, **k: None)
    assertions.assert_benchmark_report(report)
    value_checks.equal(report["summary"]["planned"], 2)
    value_checks.truthy((output / "benchmark.md").is_file())
    value_checks.equal(
        json.loads((output / "human-review.json").read_text())["status"], "pending_human_review"
    )
    value_checks.equal(
        hashlib.sha256((output / "golden-policy.json").read_bytes()).hexdigest(), dataset.sha256
    )
    errors.rejects(
        lambda: runner.run(root, output, plan, dataset, gates, notify=lambda *a, **k: None),
        expected=FileExistsError,
    )
    assertions.assert_benchmark_report(load_saved_benchmark(output / "benchmark.json"))
    faith_path = output / "case-001/faithfulness.json"
    faith_path.write_text("{}")
    errors.rejects(
        lambda: load_saved_benchmark(output / "benchmark.json"),
        expected=ValueError,
        match="checksum mismatch",
    )


@title("Benchmark dry run performs no model or application operations")
def test_dry_run(monkeypatch, tmp_path, capsys, mock_factory):
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark", "--root", str(ROOT), "--output", str(tmp_path / "new"), "--dry-run"],
    )
    monkeypatch.setattr(
        runner, "run", mock_factory(side_effect=AssertionError("Unexpected execution"))
    )
    value_checks.equal(runner.main(), 0)
    value_checks.equal(json.loads(capsys.readouterr().out)["maximum_model_calls"], 43)
    value_checks.falsy((tmp_path / "new").exists())


@title("Generation capture records teardown errors rather than treating a saved answer as success")
def test_generation_teardown_error(tmp_path, monkeypatch):
    root = tmp_path / "automation"
    samples = root / "reports/rag-samples"
    samples.mkdir(parents=True)
    directory = tmp_path / "case-001"
    directory.mkdir()
    dataset = load_golden_dataset(
        ROOT / "test_data/golden-policy.json", ROOT / "test_data/company-policy.txt"
    )
    sample_path = samples / "sample.json"
    write_sample(sample_path, _sample(dataset.cases[0], dataset))

    subprocess_run = make_subprocess_run_stub(sample_path)

    monkeypatch.setattr(runner.subprocess, "run", subprocess_run)
    result = runner.generate_sample(root, directory, "paid_leave", "qwen3.5:4b")
    value_checks.equal(result["generation_status"], "error")
    value_checks.equal(result["generation_details"][0]["message"], "Cleanup failed")
    value_checks.truthy((directory / "sample.json").is_file())


@title("Preflight failures retain every planned result as an error without generation")
def test_preflight_failure(tmp_path, benchmark_data, monkeypatch, mock_factory):
    dataset, gates, plan, _, _ = benchmark_data
    root = tmp_path / "automation"
    shutil.copytree(ROOT / "test_data", root / "test_data")
    monkeypatch.setattr(
        runner.OllamaClient, "list_models", mock_factory(side_effect=RuntimeError("Offline"))
    )
    generate = mock_factory(side_effect=AssertionError("Unexpected generation"))
    monkeypatch.setattr(runner, "generate_sample", generate)
    report = runner.run(
        root, tmp_path / "offline", plan, dataset, gates, notify=lambda *a, **k: None
    )
    value_checks.equal(report["status"], "error")
    value_checks.equal(report["summary"]["errors"], 2)
    value_checks.equal(report["summary"]["metrics"]["faithfulness"]["unavailable"], 1)
    mock_checks.not_called(generate)


@title("Two-model benchmark compares the same settings and retains per-model outcomes")
def test_two_model_summary(benchmark_data):
    dataset, gates, _, _, calibration = benchmark_data
    plan = make_plan(dataset, case_ids=["paid_leave"], models=["qwen3.5:4b", "qwen2.5:7b"])
    definition = manifest(plan, dataset, gates, ROOT / "test_data/faithfulness-controls.json")
    first, second = _row(), _row()
    second["model"] = "qwen2.5:7b"
    second["model_digest"] = "other-weights"
    first["workspace_configuration"]["chatModel"] = first["model"]
    second["workspace_configuration"]["chatModel"] = second["model"]
    report = summarize(definition, [first, second], calibration)
    assertions.assert_benchmark_report(report)
    value_checks.equal(set(report["models"]), set(plan.models))
    second["workspace_configuration"]["openAiPrompt"] = "Other prompt"
    errors.rejects(
        lambda: summarize(definition, [first, second], calibration),
        expected=ValueError,
        match="different workspace settings",
    )


@title("Offline benchmark rendering recomputes a forged passing summary without model calls")
def test_forged_top_level_summary(tmp_path, benchmark_data):
    dataset, gates, _, definition, _ = benchmark_data
    make_forged_benchmark_summary(tmp_path)
    write_sample(tmp_path / "manifest.json", definition)
    saved = {
        "manifest": definition,
        "results": [],
        "calibration": {"status": "error"},
        "status": "checks_passed",
    }
    write_sample(tmp_path / "benchmark.json", saved)
    report = load_saved_benchmark(tmp_path / "benchmark.json")
    value_checks.equal(report["status"], "error")
    value_checks.equal(report["summary"]["missing"], 2)
    errors.rejects(
        lambda: assertions.assert_benchmark_report(report),
        expected=AssertionError,
        match="did not pass",
    )
