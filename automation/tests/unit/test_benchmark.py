import copy
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

from llm_testkit import assertions
from llm_testkit.config import Settings
from llm_testkit.datasets.benchmark import make_plan, manifest
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.evaluation import benchmark as evaluation
from llm_testkit.evaluation import benchmark_runner as runner
from llm_testkit.observation.evaluation_sample import build_sample, write_sample
from llm_testkit.reporting.benchmark import markdown, review_worksheet, summarize
from llm_testkit.reporting.benchmark_evidence import load_saved_benchmark
from llm_testkit.reporting.gates import METRICS, load_quality_gates
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def benchmark_data():
    dataset = load_golden_dataset(
        ROOT / "test_data/golden-policy.json", ROOT / "test_data/company-policy.txt"
    )
    gates = load_quality_gates(ROOT / "test_data/quality-gates.json")
    plan = make_plan(dataset, case_ids=["paid_leave", "gym_missing"])
    definition = manifest(plan, dataset, gates, ROOT / "test_data/faithfulness-controls.json")
    calibration = {
        "status": "matched",
        "judge_model": plan.judge_model,
        "judge_model_digest": "judge-digest",
        "controls_sha256": definition["controls_sha256"],
        "control_ids": definition["control_ids"],
        "results": [{"status": "matched"} for _ in range(3)],
    }
    return dataset, gates, plan, definition, calibration


def _sample(case, dataset, model="qwen3.5:4b"):
    identifier = "a" * 32
    document = f"automation-{identifier}-company-policy.txt"
    context = (
        f"<document_metadata>\nsourceDocument: {document}\n</document_metadata>\n"
        + (ROOT / "test_data/company-policy.txt").read_text()
    )
    capture = {
        "schema_version": 1,
        "boundary": "ollama-sdk-chat",
        "request": {
            "model": model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": f"[LLM_TESTKIT_CAPTURE:{identifier}]\n[CONTEXT 0]:\n{context}\n[END CONTEXT 0]",
                },
                {"role": "user", "content": case.question},
            ],
        },
    }
    sample = build_sample(
        capture,
        question=case.question,
        answer=case.reference,
        reference=case.reference,
        expected_model=model,
        capture_id=identifier,
    )
    sample["metadata"] = {
        "golden_case_id": case.id,
        "golden_category": case.category,
        "golden_dataset_sha256": dataset.sha256,
        "policy_sha256": dataset.policy_sha256,
        "model_digest": "generation-digest",
        "workspace_configuration": {"openAiPrompt": "Policy only", "chatModel": model},
    }
    sample["response_sources"] = [{"title": document, "text": context}]
    return sample


def _row(case_id="paid_leave", category="multi_fact"):
    refusal = category == "missing_information"
    return {
        "case_id": case_id,
        "category": category,
        "model": "qwen3.5:4b",
        "model_digest": "generation-digest",
        "workspace_configuration": {"openAiPrompt": "Policy only"},
        "generation_status": "passed",
        "dimensions": [
            {"name": "Reviewed answer rules", "status": "passed"},
            {"name": "Document sources", "status": "passed"},
            *[
                {
                    "name": m,
                    "metric": m,
                    "status": "not_applicable" if refusal else "passed",
                    "value": 1.0,
                    "minimum": {
                        "faithfulness": 0.9,
                        "factual_correctness": 0.8,
                        "context_precision": 0.8,
                        "context_recall": 0.9,
                    }[m],
                }
                for m in sorted(METRICS)
            ],
        ],
    }


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
    [
        {"case_ids": []},
        {"case_ids": ["unknown"]},
        {"case_ids": ["gym_missing"]},
        {"case_ids": ["paid_leave", "paid_leave"]},
        {"models": []},
        {"models": ["x", "x"]},
        {"models": ["x", "y", "z"]},
        {"models": ["../bad[model]"]},
        {"max_model_calls": 42},
        {"max_model_calls": True},
        {"max_model_calls": 401},
    ],
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
    [
        "duplicate",
        "unknown",
        "category",
        "missing_metric",
        "nan",
        "minimum",
        "false_pass",
        "inapplicable",
    ],
)
@title("Benchmark refuses inconsistent case identities and metric outcomes [{param_id}]")
def test_invalid_summary(benchmark_data, change):
    _, _, _, definition, calibration = benchmark_data
    row = _row()
    rows = [row]
    if change == "duplicate":
        rows.append(copy.deepcopy(row))
    elif change == "unknown":
        row["case_id"] = "unknown"
    elif change == "category":
        row["category"] = "boundary"
    elif change == "missing_metric":
        row["dimensions"].pop()
    elif change == "nan":
        row["dimensions"][2]["value"] = float("nan")
    elif change == "minimum":
        row["dimensions"][2]["minimum"] = 0.2
    elif change == "false_pass":
        row["dimensions"][2]["value"] = 0.1
    else:
        row["dimensions"][2]["status"] = "not_applicable"
    with pytest.raises((ValueError, AssertionError)):
        summarize(definition, rows, calibration)


@title("Model comparisons reject changed prompts or generation weights")
def test_different_configurations_rejected(benchmark_data):
    _, _, _, definition, calibration = benchmark_data
    for change in ("prompt", "digest"):
        row = _row("gym_missing", "missing_information")
        if change == "prompt":
            row["workspace_configuration"]["openAiPrompt"] = "Changed prompt"
        else:
            row["model_digest"] = "changed"
        with pytest.raises(ValueError, match="changed"):
            summarize(definition, [_row(), row], calibration)


@title("Judge control mismatch prevents a benchmark from claiming acceptance")
def test_control_mismatch(benchmark_data):
    _, _, _, definition, calibration = benchmark_data
    calibration["results"][0]["status"] = "mismatch"
    report = summarize(
        definition, [_row(), _row("gym_missing", "missing_information")], calibration
    )
    assert report["status"] == "error"
    assert report["judge_controls_matched"] is False


def _mock_metrics(monkeypatch, case, dataset, sample_path):
    checksum = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    verdict = {"statement": case.reference, "verdict": 1, "reason": "Supported"}
    faith = {
        "schema_version": 1,
        "metric": "faithfulness",
        "status": "completed",
        "sample_sha256": checksum,
        "judge_model": "qwen3.5:4b",
        "judge_model_digest": "judge-digest",
        "judge_configuration": {"think": False},
        "ragas_version": "0.4.3",
        "created_at": "now",
        "result": {"value": 1.0, "statements": [case.reference], "verdicts": [verdict]},
        "judge_calls": [
            {"output": {"statements": [case.reference]}},
            {"output": {"statements": [verdict]}},
        ],
    }
    from llm_testkit.evaluation.correctness import METRIC_CONFIGURATION

    correct = {
        **faith,
        "metric": "factual_correctness",
        "response_origin": "application_sample",
        "golden_dataset_sha256": dataset.sha256,
        "golden_case_id": case.id,
        "metric_configuration": METRIC_CONFIGURATION,
        "question": case.question,
        "reference": case.reference,
        "response": case.reference,
        "reference_sha256": hashlib.sha256(case.reference.encode()).hexdigest(),
        "result": {
            "value": 1.0,
            "response_claims": [case.reference],
            "reference_claims": [case.reference],
            "response_verdicts": [verdict],
            "reference_verdicts": [verdict],
        },
        "judge_calls": [
            {"output": {"claims": [case.reference]}},
            {"output": {"statements": [verdict]}},
            {"output": {"claims": [case.reference]}},
            {"output": {"statements": [verdict]}},
        ],
    }
    precision = {"verdict": 1, "reason": "Relevant"}
    recall = {"statement": case.reference, "attributed": 1, "reason": "Supported"}
    relevance = {
        **correct,
        "metric": "context_relevance",
        "result": {
            "context_precision": 1 / (1 + 1e-10),
            "context_recall": 1.0,
            "precision_verdicts": [precision],
            "recall_classifications": [recall],
        },
        "precision_calls": [{"output": precision}],
        "recall_calls": [{"output": {"classifications": [recall]}}],
    }
    relevance.pop("judge_calls")
    for function, result in (
        ("evaluate_sample_report", faith),
        ("evaluate_correctness_report", correct),
        ("evaluate_relevance_report", relevance),
    ):
        monkeypatch.setattr(evaluation, function, Mock(return_value=result))


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

    def calibrate(path, controls, settings, model, digest):
        calibrated = copy.deepcopy(calibration)
        calibrated["sample_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        catalog = json.loads(controls.read_text())["cases"]
        calibrated["results"] = []
        for identifier in calibrated["control_ids"]:
            control = next(c for c in catalog if c["id"] == identifier)
            statements = control["response"].split(". ")
            verdicts = [
                {"statement": s, "verdict": c["verdict"], "reason": "Control label"}
                for s, c in zip(statements, control["claims"], strict=True)
            ]
            calibrated["results"].append(
                {
                    "status": "matched",
                    "control": control,
                    "result": {
                        "value": control["expected_score"],
                        "statements": statements,
                        "verdicts": verdicts,
                    },
                    "judge_calls": [
                        {"output": {"statements": statements}},
                        {"output": {"statements": verdicts}},
                    ],
                }
            )
        return calibrated

    monkeypatch.setattr(runner, "calibrate", calibrate)

    def generate(root, directory, identifier, model):
        case = next(c for c in dataset.cases if c.id == identifier)
        sample = _sample(case, dataset, model)
        sample["metadata"]["model_digest"] = "judge-digest"
        write_sample(directory / "sample.json", sample)
        _mock_metrics(monkeypatch, case, dataset, directory / "sample.json")
        junit = directory / "generation.xml"
        junit.write_text(
            f'<testsuite><testcase classname="tests.test_golden_rag" name="test_golden_policy_answer[{identifier}-{model}]"/></testsuite>'
        )
        return {
            "generation_status": "passed",
            "generation_junit_sha256": hashlib.sha256(junit.read_bytes()).hexdigest(),
        }

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

    def subprocess_run(command, **kwargs):
        junit_path = Path(command[command.index("--junitxml") + 1])
        junit_path.write_text(
            f'<testsuite><testcase classname="tests.test_golden_rag" name="test_golden_policy_answer[paid_leave-qwen3.5:4b]"><properties><property name="evaluation_sample" value="{sample_path}"/></properties><error message="Cleanup failed"/></testcase></testsuite>'
        )
        return Mock(returncode=1)

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
    for name in (
        "golden-policy.json",
        "quality-gates.json",
        "faithfulness-controls.json",
        "company-policy.txt",
    ):
        (tmp_path / name).write_bytes((ROOT / "test_data" / name).read_bytes())
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
