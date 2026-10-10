import hashlib
import json
import shutil

import pytest

from llm_testkit import assertions
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.evaluation import benchmark_runner as runner
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.benchmark_evidence import load_saved_benchmark
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import mocks as mock_checks
from test_support.assertions import values as value_checks
from test_support.builders.benchmark import (
        make_benchmark_sample, make_calibrate_stub, make_forged_benchmark_summary, make_forged_saved_report,
        make_generate_stub, make_judge_model_catalog, make_subprocess_run_stub)
from test_support.data import common as case_data
from test_support.data.benchmark import ANSWER_DURATION_SECONDS, ROOT
from test_support.fixtures.unit_benchmark import benchmark_data as benchmark_data
from test_support.fixtures.unit_benchmark import measured_chat as measured_chat

pytestmark = pytest.mark.unit


@title("Answer timing covers the chat request and is recorded in JUnit metadata")
def test_chat_duration_is_measured(measured_chat):
    chat, client, record_property, clock = measured_chat
    result = chat("Leave?", "Reference")
    value_checks.identical(result, client.chat.return_value)
    mock_checks.called_once_with(record_property, "answer_request_seconds", ANSWER_DURATION_SECONDS)
    value_checks.equal(clock.call_count, 2)


@title("Benchmark runner preserves its full matrix, snapshots and human review worksheet")
def test_runner_pipeline(tmp_path, benchmark_data, monkeypatch, mock_factory):
    dataset, gates, plan, _, calibration = benchmark_data
    root = tmp_path / "automation"
    shutil.copytree(ROOT / "test_data", root / "test_data")
    catalog = mock_factory(status_code=200)
    catalog.json.return_value = make_judge_model_catalog()
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
    value_checks.equal(json.loads((output / "human-review.json").read_text())["status"], "pending_human_review")
    value_checks.equal(hashlib.sha256((output / "golden-policy.json").read_bytes()).hexdigest(), dataset.sha256)
    errors.rejects(
            lambda: runner.run(root, output, plan, dataset, gates, notify=lambda *a, **k: None),
            expected=FileExistsError)
    assertions.assert_benchmark_report(load_saved_benchmark(output / "benchmark.json"))
    faith_path = output / "case-001/faithfulness.json"
    faith_path.write_text("{}")
    errors.rejects(
            lambda: load_saved_benchmark(output / "benchmark.json"), expected=ValueError, match="checksum mismatch")


@title("Benchmark dry run performs no model or application operations")
def test_dry_run(monkeypatch, tmp_path, capsys, mock_factory, failure_factory):
    monkeypatch.setattr("sys.argv", ["benchmark", "--root", str(ROOT), "--output", str(tmp_path / "new"), "--dry-run"])
    monkeypatch.setattr(
            runner, "run", mock_factory(side_effect=failure_factory(AssertionError, "Unexpected execution")))
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
    dataset = load_golden_dataset(ROOT / "test_data/golden-policy.json", ROOT / "test_data/company-policy.txt")
    sample_path = samples / case_data.SAMPLE_FILE_NAME
    write_sample(sample_path, make_benchmark_sample(dataset.cases[0]))

    subprocess_run = make_subprocess_run_stub(sample_path)

    monkeypatch.setattr(runner.subprocess, "run", subprocess_run)
    result = runner.generate_sample(root, directory, "paid_leave", "qwen3.5:4b")
    value_checks.equal(result["generation_status"], "error")
    value_checks.equal(result["generation_details"][0]["message"], "Cleanup failed")
    value_checks.truthy((directory / case_data.SAMPLE_FILE_NAME).is_file())


@title("Preflight failures retain every planned result as an error without generation")
def test_preflight_failure(tmp_path, benchmark_data, monkeypatch, mock_factory, failure_factory):
    dataset, gates, plan, _, _ = benchmark_data
    root = tmp_path / "automation"
    shutil.copytree(ROOT / "test_data", root / "test_data")
    monkeypatch.setattr(
            runner.OllamaClient, "list_models", mock_factory(side_effect=failure_factory(RuntimeError, "Offline")))
    generate = mock_factory(side_effect=failure_factory(AssertionError, "Unexpected generation"))
    monkeypatch.setattr(runner, "generate_sample", generate)
    report = runner.run(root, tmp_path / "offline", plan, dataset, gates, notify=lambda *a, **k: None)
    value_checks.equal(report["status"], "error")
    value_checks.equal(report["summary"]["errors"], 2)
    value_checks.equal(report["summary"]["metrics"]["faithfulness"]["unavailable"], 1)
    mock_checks.not_called(generate)


@title("Offline benchmark rendering recomputes a forged passing summary without model calls")
def test_forged_top_level_summary(tmp_path, benchmark_data):
    dataset, gates, _, definition, _ = benchmark_data
    make_forged_benchmark_summary(tmp_path)
    write_sample(tmp_path / "manifest.json", definition)
    saved = make_forged_saved_report(definition)
    write_sample(tmp_path / "benchmark.json", saved)
    report = load_saved_benchmark(tmp_path / "benchmark.json")
    value_checks.equal(report["status"], "error")
    value_checks.equal(report["summary"]["missing"], 2)
    errors.rejects(lambda: assertions.assert_benchmark_report(report), expected=AssertionError, match="did not pass")
