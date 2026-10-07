import hashlib
import json
import shutil
from unittest.mock import Mock

import pytest

from llm_testkit import assertions
from llm_testkit.config import Settings
from llm_testkit.datasets.benchmark import make_plan, manifest
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.evaluation import benchmark as evaluation
from llm_testkit.evaluation import benchmark_runner as runner
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.benchmark import markdown, review_worksheet, summarize
from llm_testkit.reporting.benchmark_evidence import load_saved_benchmark
from llm_testkit.reporting.steps import title
from test_support.builders.benchmark import (
    _mock_metrics,
    _row,
    _sample,
    make_calibrate_stub,
    make_forged_benchmark_summary,
    make_generate_stub,
    make_subprocess_run_stub,
    mutate_benchmark_configuration,
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
    assert make_plan(dataset).maximum_calls == 43
    assert (
        make_plan(dataset, models=["qwen3.5:4b", "qwen2.5:7b"], max_model_calls=80).maximum_calls
        == 80
    )


@pytest.mark.parametrize(
    "options",
    INVALID_PLAN_OPTIONS_CASES,
)
@title("Benchmark rejects invalid or over-budget matrices before model calls [{param_id}]")
def test_invalid_plan(benchmark_data, options):
    with pytest.raises(ValueError):
        make_plan(benchmark_data[0], **options)


@title("Benchmark summary excludes refusal metrics explicitly and uses the planned denominator")
def test_summary_and_missing_rows(benchmark_data):
    _, _, _, definition, calibration = benchmark_data
    report = summarize(definition, [_row()], calibration)
    assert report["status"] == "error"
    assert report["summary"]["pass_rate"] == 0.5
    assert report["summary"]["missing"] == 1
    assert report["summary"]["metrics"]["context_recall"]["not_applicable"] == 1
    assert "1 / 1" in markdown(report)
    complete = summarize(
        definition, [_row(), _row("gym_missing", "missing_information")], calibration
    )
    assertions.assert_benchmark_report(complete)
    assert complete["categories"]["missing_information"]["metrics"]["faithfulness"]["mean"] is None
    assert review_worksheet(complete)["status"] == "pending_human_review"


@title("Failed answer checks and judge errors cannot be hidden by perfect mean scores")
def test_failures_remain_visible(benchmark_data):
    _, _, _, definition, calibration = benchmark_data
    row = _row()
    row["dimensions"][0].update(status="failed", error="Missing annual allowance")
    report = summarize(definition, [row, _row("gym_missing", "missing_information")], calibration)
    assert report["status"] == "failed"
    assert report["summary"]["metrics"]["faithfulness"]["mean"] == 1
    row["dimensions"][2] = {
        "name": "context_precision",
        "metric": "context_precision",
        "status": "error",
    }
    report = summarize(definition, [row], calibration)
    assert report["summary"]["metrics"]["context_precision"]["unavailable"] == 1
    assert report["summary"]["metrics"]["context_precision"]["mean"] is None


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
    with pytest.raises((ValueError, AssertionError)):
        summarize(definition, rows, calibration)


@title("Model comparisons reject changed prompts or generation weights")
def test_different_configurations_rejected(benchmark_data):
    _, _, _, definition, calibration = benchmark_data
    mutate_benchmark_configuration(calibration, definition)


@title("Judge control mismatch prevents a benchmark from claiming acceptance")
def test_control_mismatch(benchmark_data):
    _, _, _, definition, calibration = benchmark_data
    calibration["results"][0]["status"] = "mismatch"
    report = summarize(
        definition, [_row(), _row("gym_missing", "missing_information")], calibration
    )
    assert report["status"] == "error"
    assert report["judge_controls_matched"] is False


@title("All four benchmark metrics validate original evidence and preserve independent failures")
def test_case_pipeline(tmp_path, benchmark_data, monkeypatch):
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
        settings=Settings(),
        minima=gates["minimum_scores"],
    )
    row = evaluation.evaluate_case(sample_path, **kwargs)
    assert len(row["dimensions"]) == 6
    assert all(d["status"] == "passed" for d in row["dimensions"])
    assert row["judge_calls"] == 8
    assert len(row["evidence_sha256"]) == 3
    other = tmp_path / "failed"
    other.mkdir()
    evaluation.evaluate_correctness_report.return_value["sample_sha256"] = "wrong"
    row = evaluation.evaluate_case(sample_path, **{**kwargs, "directory": other})
    assert (
        next(d for d in row["dimensions"] if d.get("metric") == "factual_correctness")["status"]
        == "error"
    )
    assert (
        next(d for d in row["dimensions"] if d.get("metric") == "context_recall")["status"]
        == "passed"
    )


@title("Refusal cases run reviewed checks without invoking a semantic judge")
def test_refusal_case_no_judge(tmp_path, benchmark_data, monkeypatch):
    dataset, gates, _, _, _ = benchmark_data
    case = next(c for c in dataset.cases if c.id == "gym_missing")
    path = tmp_path / "sample.json"
    write_sample(path, _sample(case, dataset))
    judge = Mock(side_effect=AssertionError("Unexpected judge call"))
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
        settings=Settings(),
        minima=gates["minimum_scores"],
    )
    assert row["judge_calls"] == 0
    assert sum(d["status"] == "not_applicable" for d in row["dimensions"]) == 4
    judge.assert_not_called()


@title("Benchmark runner preserves its full matrix, snapshots and human review worksheet")
def test_runner_pipeline(tmp_path, benchmark_data, monkeypatch):
    dataset, gates, plan, _, calibration = benchmark_data
    root = tmp_path / "automation"
    shutil.copytree(ROOT / "test_data", root / "test_data")
    catalog = Mock(status_code=200)
    catalog.json.return_value = {"models": [{"name": "qwen3.5:4b", "digest": "judge-digest"}]}
    monkeypatch.setattr(runner.OllamaClient, "list_models", lambda _: catalog)

    calibrate = make_calibrate_stub(calibration)

    monkeypatch.setattr(runner, "calibrate", calibrate)

    generate = make_generate_stub(dataset, monkeypatch)

    monkeypatch.setattr(runner, "generate_sample", generate)
    output = tmp_path / "run"
    report = runner.run(root, output, plan, dataset, gates, notify=lambda *a, **k: None)
    assertions.assert_benchmark_report(report)
    assert report["summary"]["planned"] == 2
    assert (output / "benchmark.md").is_file()
    assert (
        json.loads((output / "human-review.json").read_text())["status"] == "pending_human_review"
    )
    assert (
        hashlib.sha256((output / "golden-policy.json").read_bytes()).hexdigest() == dataset.sha256
    )
    with pytest.raises(FileExistsError):
        runner.run(root, output, plan, dataset, gates, notify=lambda *a, **k: None)
    assertions.assert_benchmark_report(load_saved_benchmark(output / "benchmark.json"))
    faith_path = output / "case-001/faithfulness.json"
    faith_path.write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_saved_benchmark(output / "benchmark.json")


@title("Benchmark dry run performs no model or application operations")
def test_dry_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["benchmark", "--root", str(ROOT), "--output", str(tmp_path / "new"), "--dry-run"],
    )
    monkeypatch.setattr(runner, "run", Mock(side_effect=AssertionError("Unexpected execution")))
    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out)["maximum_model_calls"] == 43
    assert not (tmp_path / "new").exists()


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
    assert result["generation_status"] == "error"
    assert result["generation_details"][0]["message"] == "Cleanup failed"
    assert (directory / "sample.json").is_file()


@title("Preflight failures retain every planned result as an error without generation")
def test_preflight_failure(tmp_path, benchmark_data, monkeypatch):
    dataset, gates, plan, _, _ = benchmark_data
    root = tmp_path / "automation"
    shutil.copytree(ROOT / "test_data", root / "test_data")
    monkeypatch.setattr(
        runner.OllamaClient, "list_models", Mock(side_effect=RuntimeError("Offline"))
    )
    generate = Mock(side_effect=AssertionError("Unexpected generation"))
    monkeypatch.setattr(runner, "generate_sample", generate)
    report = runner.run(
        root, tmp_path / "offline", plan, dataset, gates, notify=lambda *a, **k: None
    )
    assert report["status"] == "error"
    assert report["summary"]["errors"] == 2
    assert report["summary"]["metrics"]["faithfulness"]["unavailable"] == 1
    generate.assert_not_called()


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
    assert set(report["models"]) == set(plan.models)
    second["workspace_configuration"]["openAiPrompt"] = "Other prompt"
    with pytest.raises(ValueError, match="different workspace settings"):
        summarize(definition, [first, second], calibration)


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
    assert report["status"] == "error"
    assert report["summary"]["missing"] == 2
    with pytest.raises(AssertionError, match="did not pass"):
        assertions.assert_benchmark_report(report)
