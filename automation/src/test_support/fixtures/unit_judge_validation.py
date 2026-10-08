"""Scoped source catalogs, judge adapters and invalid-evidence scenarios."""

import asyncio
import json
import shutil
from copy import deepcopy
from unittest.mock import Mock

import pytest

from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.datasets.judge_controls import load_judge_controls, maximum_calls
from llm_testkit.evaluation import benchmark as benchmark_evaluation
from llm_testkit.evaluation import benchmark_runner
from llm_testkit.evaluation.judge_validation import evaluate_judge_controls, main
from test_support.builders.benchmark import make_calibrate_stub, make_generate_stub, make_judge_model_catalog
from test_support.builders.correctness import append_judge_responses
from test_support.builders.judge_validation import SavedReviewScenario, ValidationBatch, control_outputs, mutate_catalog
from test_support.builders.optional import load_ollama_judge
from test_support.data import common as case_data
from test_support.data.judge_validation import ROOT


@pytest.fixture
def judge_catalog():
    directory = ROOT / "test_data"
    dataset = load_golden_dataset(directory / "golden-policy.json", directory / "company-policy.txt")
    return load_judge_controls(directory / "judge-validation.json", dataset, directory / "company-policy.txt")


@pytest.fixture
def judge_control(judge_catalog, control_id):
    return deepcopy(next(case for case in judge_catalog.cases if case["id"] == control_id))


@pytest.fixture
def real_metric_control(judge_control):
    responses = []
    append_judge_responses(control_outputs(judge_control), responses)
    client = Mock()
    client.structured_chat.side_effect = responses
    judge = load_ollama_judge()(client, case_data.TEST_MODEL, max_calls=maximum_calls(judge_control))
    return lambda: asyncio.run(evaluate_judge_controls([judge_control], lambda budget: judge))


@pytest.fixture
def invalid_judge_catalog(tmp_path, change):
    directory = ROOT / "test_data"
    data = json.loads((directory / "judge-validation.json").read_text())
    mutate_catalog(data, change)
    destination = tmp_path / "invalid-controls.json"
    destination.write_text(json.dumps(data))
    dataset = load_golden_dataset(directory / "golden-policy.json", directory / "company-policy.txt")
    return lambda: load_judge_controls(destination, dataset, directory / "company-policy.txt")


@pytest.fixture
def failed_validation_batch(judge_catalog):
    controls = deepcopy(list(judge_catalog.cases[:4]))
    factory = Mock(side_effect=lambda budget: Mock(calls=[], max_calls=budget))

    async def scorer(control, judge):
        if control["id"] == controls[2]["id"]:
            judge.calls.append({"error": "truncated"})
            raise ValueError("Truncated judge evidence")
        outputs = control_outputs(control)
        value = control["expected_score"]
        if control["id"] == controls[1]["id"]:
            outputs[1]["statements"][0]["verdict"] = 1
            value = 1.0
        judge.calls.extend({"output": output} for output in outputs)
        return value

    return ValidationBatch(controls, factory, scorer)


@pytest.fixture
def reversed_recall_labels(judge_catalog):
    control = deepcopy(judge_catalog.cases[-1])
    rows = deepcopy(control_outputs(control)[0]["classifications"])
    rows[0]["verdict"], rows[1]["verdict"] = 0, 1
    return control, {"value": 0.5, "reference": rows}


@pytest.fixture
def corrupt_precision_evidence(judge_catalog):
    control = deepcopy(judge_catalog.cases[4])
    calls = [{"output": output} for output in control_outputs(control)]
    return control, calls


@pytest.fixture
def saved_judge_review(tmp_path, monkeypatch, benchmark_data):
    dataset, gates, plan, _, calibration = benchmark_data
    root = tmp_path / "automation"
    shutil.copytree(ROOT / "test_data", root / "test_data")
    inventory = Mock(status_code=200)
    inventory.json.return_value = make_judge_model_catalog()
    monkeypatch.setattr(benchmark_runner.OllamaClient, "list_models", lambda _: inventory)
    monkeypatch.setattr(benchmark_runner, "calibrate", make_calibrate_stub(calibration))
    generate = make_generate_stub(dataset, monkeypatch)

    def disputed_generation(*args):
        result = generate(*args)
        evidence = deepcopy(benchmark_evaluation.evaluate_correctness_report.return_value)
        evidence["result"]["value"] = 0.0
        for direction in ("response", "reference"):
            for row in evidence["result"][f"{direction}_verdicts"]:
                row["verdict"] = 0
        for call in evidence["judge_calls"]:
            for row in call["output"].get("statements", []):
                row["verdict"] = 0
        benchmark_evaluation.evaluate_correctness_report.return_value = evidence
        return result

    monkeypatch.setattr(benchmark_runner, "generate_sample", disputed_generation)
    destination = tmp_path / "benchmark"
    benchmark_runner.run(root, destination, plan, dataset, gates, notify=lambda *args, **kwargs: None)
    return SavedReviewScenario(destination / "benchmark.json", destination / "case-001/faithfulness.json")


@pytest.fixture
def judge_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    transport = Mock(side_effect=AssertionError("Unexpected network access"))
    monkeypatch.setattr("llm_testkit.evaluation.judge_validation.HttpClient", transport)

    def invoke(arguments):
        monkeypatch.setattr("sys.argv", ["judge_validation", *arguments])
        return main()

    output = tmp_path / "existing.json"
    output.write_text("original")
    return invoke, transport, output
