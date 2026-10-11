"""Benchmark runner: isolated generation through pytest, judge controls, the serial matrix run and the CLI."""

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from llm_testkit.config import Settings
from llm_testkit.datasets.benchmark import CONTROL_IDS, make_plan
from llm_testkit.evaluation import benchmark_runner as runner
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.steps import title
from test_support.builders.benchmark import (
        JUDGE_DIGEST, MODEL, QUALITY_GATES, make_benchmark_sample, make_calibrate_stub, make_calibration,
        make_definition, make_generate_stub)
from test_support.builders.calibration import CONTROLS_FILE
from test_support.builders.golden import GOLDEN_DATASET, PAID_LEAVE, TEST_DATA, make_case_sample
from test_support.builders.ollama import model_catalog

pytestmark = pytest.mark.unit

NODE = "test_golden_policy_answer[paid_leave-qwen3.5:4b]"


@pytest.fixture
def automation_root(tmp_path, monkeypatch):
    """A minimal automation checkout with the reviewed test data, and no credentials in the environment."""
    monkeypatch.delenv("ANYTHINGLLM_API_KEY", raising=False)
    monkeypatch.delenv("ANYTHINGLLM_API_KEY_FILE", raising=False)
    root = tmp_path / "automation"
    shutil.copytree(TEST_DATA, root / "test_data")
    (root / "tests").mkdir()
    (root / "tests/test_golden_rag.py").write_text("")
    (root / "reports/rag-samples").mkdir(parents=True)
    return root


def junit_xml(*, name: str = NODE, properties: dict[str, str] | None = None, outcome: str = "") -> str:
    rows = "".join(f'<property name="{key}" value="{value}"/>' for key, value in (properties or {}).items())
    return (
            f'<testsuite><testcase classname="tests.test_golden_rag" name="{name}">'
            f"<properties>{rows}</properties>{outcome}</testcase></testsuite>")


@pytest.fixture
def generation(automation_root, monkeypatch):
    """generate_sample with pytest replaced by a double that writes a chosen JUnit report and exit code."""
    sample = make_benchmark_sample(PAID_LEAVE)
    sample_path = automation_root / "reports/rag-samples/sample.json"
    write_sample(sample_path, sample)
    directory = automation_root.parent / "case-001"
    directory.mkdir()
    state = SimpleNamespace(
            root=automation_root, directory=directory, sample=sample, sample_path=sample_path, returncode=0,
            junit=junit_xml(properties={"evaluation_sample": str(sample_path)}), log="pytest output")

    def run_pytest(command, **kwargs):
        Path(command[command.index("--junitxml") + 1]).write_text(state.junit)
        kwargs["stdout"].write(state.log)
        return Mock(returncode=state.returncode)

    state.subprocess_run = Mock(side_effect=run_pytest)
    monkeypatch.setattr(runner.subprocess, "run", state.subprocess_run)
    state.generate = lambda: runner.generate_sample(automation_root, directory, "paid_leave", MODEL)
    return state


@title("Generation runs the golden test node in a child pytest with capture enabled and plugin autoload off")
def test_generation_command(generation):
    generation.generate()

    command = generation.subprocess_run.call_args.args[0]
    options = generation.subprocess_run.call_args.kwargs
    assert command == [
            sys.executable, "-m", "pytest", f"tests/test_golden_rag.py::{NODE}", "--run-golden", "--capture-rag",
            "--rag-model", MODEL, "--rag-repeat", "1", "--junitxml",
            str(generation.directory / "generation.xml"), "-o",
            "addopts=-ra --strict-markers --strict-config --import-mode=importlib", "-q"]
    assert options["cwd"] == generation.root
    assert (options["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"], options["env"]["PYTEST_ADDOPTS"]) == ("1", "")
    assert (options["stderr"], options["check"]) == (subprocess.STDOUT, False)
    assert options["stdout"].name == str(generation.directory / "generation.log")


@title("A passing generation saves its captured sample and reports the JUnit outcome and checksum")
def test_generation_saves_sample(generation):
    result = generation.generate()

    xml = (generation.directory / "generation.xml").read_bytes()
    assert result == {
            "generation_status": "passed",
            "generation_exit_code": 0,
            "generation_junit_sha256": hashlib.sha256(xml).hexdigest(),
            "generation_details": []}
    assert json.loads((generation.directory / "sample.json").read_text()) == generation.sample
    assert (generation.directory / "generation.log").read_text() == "pytest output"


@pytest.mark.parametrize(("outcome", "status"), [
        pytest.param('<failure message="Missing annual allowance"/>', "failed", id="answer-failed"),
        pytest.param('<error message="Cleanup failed"/>', "error", id="teardown-error")])
@title("A failed answer or teardown error is reported with its details rather than treated as success [{param_id}]")
def test_generation_reports_failure(generation, outcome, status):
    generation.returncode = 1
    generation.junit = junit_xml(properties={"evaluation_sample": str(generation.sample_path)}, outcome=outcome)

    result = generation.generate()

    assert (result["generation_status"], result["generation_exit_code"]) == (status, 1)
    assert result["generation_details"][0]["kind"] == status
    assert (generation.directory / "sample.json").is_file()


@title("The API key is redacted from the generation log and JUnit report before they are hashed")
def test_generation_redacts_api_key(generation):
    (generation.root.parent / ".runtime").mkdir()
    (generation.root.parent / ".runtime/anythingllm-api-key").write_text("secret-key\n")
    generation.log = "Authorization: Bearer secret-key"
    generation.junit = junit_xml(properties={"evaluation_sample": str(generation.sample_path), "key": "secret-key"})

    result = generation.generate()

    log = (generation.directory / "generation.log").read_text()
    xml = (generation.directory / "generation.xml").read_bytes()
    assert log == "Authorization: Bearer [REDACTED]"
    assert b"secret-key" not in xml and b"[REDACTED]" in xml
    assert result["generation_junit_sha256"] == hashlib.sha256(xml).hexdigest()


def two_cases(state):
    return junit_xml(properties={
            "evaluation_sample": str(state.sample_path)}).replace(
            "</testsuite>", '<testcase classname="tests.test_golden_rag" name="other"/></testsuite>')


@pytest.mark.parametrize(("prepare", "message"), [
        pytest.param(lambda s: setattr(s, "junit", two_cases(s)), "exactly one JUnit case", id="two-cases"),
        pytest.param(
        lambda s: setattr(s, "junit", junit_xml(name="other")), "identity/metadata mismatch", id="other-node"),
        pytest.param(
        lambda s: setattr(
        s, "junit",
        junit_xml(properties={
        "evaluation_sample": str(s.sample_path)}).replace(
        "</properties>", '<property name="evaluation_sample" value="other"/></properties>')),
        "identity/metadata mismatch", id="conflicting-metadata"),
        pytest.param(
        lambda s: setattr(s, "junit", junit_xml(outcome="<failure/>")), "disagrees with its JUnit",
        id="exit-0-but-failed"),
        pytest.param(lambda s: setattr(s, "returncode", 1), "disagrees with its JUnit", id="exit-1-but-passed"),
        pytest.param(lambda s: setattr(s, "returncode", 2), "disagrees with its JUnit", id="usage-error-exit"),
        pytest.param(
        lambda s: setattr(s, "junit", junit_xml()), "no captured sample; JUnit status=passed", id="no-sample"),
        pytest.param(
        lambda s: setattr(s, "junit", junit_xml(properties={"evaluation_sample": str(s.root / "x.json")})),
        "no captured sample", id="sample-outside-reports"),
        pytest.param(lambda s: s.sample_path.unlink(), "no captured sample", id="sample-file-missing")])
@title("Generation evidence that is ambiguous, inconsistent or lacks a captured sample is rejected [{param_id}]")
def test_generation_rejects_inconsistent_evidence(generation, prepare, message):
    prepare(generation)

    with pytest.raises(ValueError, match=message):
        generation.generate()


@pytest.mark.parametrize(
        "junit_timing", [{}, {
        "answer_request_seconds": "9.0"}], ids=["not-in-junit", "different-in-junit"])
@title("A sample's answer duration must equal the duration recorded in the generation JUnit [{param_id}]")
def test_generation_rejects_inconsistent_duration(generation, junit_timing):
    generation.sample["metadata"]["answer_request_seconds"] = 12.5
    generation.sample_path.write_text(json.dumps(generation.sample))
    generation.junit = junit_xml(properties={"evaluation_sample": str(generation.sample_path), **junit_timing})

    with pytest.raises(ValueError, match="duration differs between sample and generation JUnit"):
        generation.generate()


@title("A matching answer duration in the sample and JUnit is accepted")
def test_generation_accepts_matching_duration(generation):
    generation.sample["metadata"]["answer_request_seconds"] = 12.5
    generation.sample_path.write_text(json.dumps(generation.sample))
    generation.junit = junit_xml(
            properties={
            "evaluation_sample": str(generation.sample_path),
            "answer_request_seconds": "12.5"})

    assert generation.generate()["generation_status"] == "passed"


@title("A captured sample without benchmark metadata is still saved")
def test_generation_accepts_sample_without_metadata(generation):
    generation.sample_path.write_text(json.dumps(make_case_sample(PAID_LEAVE, model=MODEL)))

    assert generation.generate()["generation_status"] == "passed"


@title("An existing generation log is never overwritten")
def test_generation_refuses_existing_log(generation):
    (generation.directory / "generation.log").write_text("earlier run")

    with pytest.raises(FileExistsError):
        generation.generate()


@pytest.fixture
def judge_controls(tmp_path, monkeypatch):
    """calibrate with the Ollama transport and the control runner replaced by doubles."""
    pytest.importorskip("ragas")
    sample_path = tmp_path / "sample.json"
    write_sample(sample_path, make_case_sample(PAID_LEAVE))
    http = MagicMock()
    http.__enter__.return_value = http
    state = SimpleNamespace(
            sample_path=sample_path, http=http, http_class=Mock(return_value=http), ollama=Mock(),
            judge_class=Mock(return_value=Mock()), evaluate=AsyncMock(return_value=[{
            "status": "matched"}] * 3))
    monkeypatch.setattr(runner, "HttpClient", state.http_class)
    monkeypatch.setattr(runner, "OllamaClient", Mock(return_value=state.ollama))
    monkeypatch.setattr("llm_testkit.evaluation.ollama_judge.OllamaJudge", state.judge_class)
    monkeypatch.setattr(runner, "evaluate_controls", state.evaluate)
    state.calibrate = lambda: runner.calibrate(sample_path, CONTROLS_FILE, Settings(), MODEL, JUDGE_DIGEST)
    return state


@title("Judge controls run the three declared controls in declared order with fresh judges and record provenance")
def test_calibrate_runs_declared_controls(judge_controls):
    report = judge_controls.calibrate()

    assert report["status"] == "matched"
    assert (report["judge_model"], report["judge_model_digest"]) == (MODEL, JUDGE_DIGEST)
    assert report["controls_sha256"] == hashlib.sha256(CONTROLS_FILE.read_bytes()).hexdigest()
    assert report["control_ids"] == list(CONTROL_IDS)
    assert report["sample_sha256"] == hashlib.sha256(judge_controls.sample_path.read_bytes()).hexdigest()
    assert report["results"] == [{"status": "matched"}] * 3
    sample, cases, factory = judge_controls.evaluate.await_args.args
    assert sample == json.loads(judge_controls.sample_path.read_text())
    assert [case["id"] for case in cases] == list(CONTROL_IDS)
    factory()
    judge_controls.judge_class.assert_called_once_with(judge_controls.ollama, MODEL, Settings().llm_timeout)
    runner.OllamaClient.assert_called_once_with(judge_controls.http)
    judge_controls.http_class.assert_called_once_with(Settings().ollama_base_url, Settings().http_timeout)
    judge_controls.http.__exit__.assert_called_once()


@pytest.mark.parametrize(("statuses", "status"), [
        pytest.param(["matched", "mismatch", "matched"], "mismatch", id="mismatch"),
        pytest.param(["mismatch", "error", "matched"], "error", id="error-wins")])
@title("Any control error makes the check an error; otherwise any mismatch makes it a mismatch [{param_id}]")
def test_calibrate_aggregates_control_outcomes(judge_controls, statuses, status):
    judge_controls.evaluate.return_value = [{"status": s} for s in statuses]

    assert judge_controls.calibrate()["status"] == status


@title("A sample without the controls' policy context is an error before any judge call")
def test_calibrate_rejects_unrelated_sample(judge_controls):
    judge_controls.sample_path.unlink()
    write_sample(judge_controls.sample_path, make_case_sample(PAID_LEAVE, contexts=["Unrelated document"]))

    report = judge_controls.calibrate()

    assert (report["status"], report["results"]) == ("error", [])
    assert report["error"].startswith("ValueError: Captured context does not contain the policy")
    judge_controls.evaluate.assert_not_awaited()


@pytest.fixture
def benchmark_run(automation_root, monkeypatch):
    """run() with the model catalog, judge controls and generation replaced by deterministic doubles."""
    definition = make_definition()
    state = SimpleNamespace(root=automation_root, output=automation_root.parent / "run", notify=Mock())
    state.catalog = model_catalog((MODEL, JUDGE_DIGEST))
    monkeypatch.setattr(runner.OllamaClient, "list_models", lambda _: state.catalog)
    state.calibrate = Mock(side_effect=make_calibrate_stub(make_calibration(definition)))
    monkeypatch.setattr(runner, "calibrate", state.calibrate)
    state.generate = Mock(side_effect=make_generate_stub(GOLDEN_DATASET, monkeypatch))
    monkeypatch.setattr(runner, "generate_sample", state.generate)
    state.plan = make_plan(GOLDEN_DATASET, case_ids=["gym_missing", "paid_leave"])
    state.run = lambda: runner.run(
            automation_root, state.output, state.plan, GOLDEN_DATASET, QUALITY_GATES, notify=state.notify)
    return state


@title("A run anchors judge controls on paid leave first, then saves every case, the summary and a review sheet")
def test_run_saves_complete_benchmark(benchmark_run):
    report = benchmark_run.run()

    output = benchmark_run.output
    assert report["status"] == "checks_passed"
    assert [call.args[2] for call in benchmark_run.generate.call_args_list] == ["paid_leave", "gym_missing"]
    assert [call.args[0] for call in benchmark_run.notify.call_args_list
            ] == ["[1/2] paid_leave / qwen3.5:4b", "[2/2] gym_missing / qwen3.5:4b"]
    assert all(call.kwargs == {"flush": True} for call in benchmark_run.notify.call_args_list)
    benchmark_run.calibrate.assert_called_once()
    assert benchmark_run.calibrate.call_args.args[0] == output / "case-001/sample.json"
    assert json.loads((output / "judge-controls.json").read_text())["status"] == "matched"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["created_at"] and manifest["expected_rows"] == report["manifest"]["expected_rows"]
    for name in ("golden-policy.json", "quality-gates.json", "faithfulness-controls.json", "company-policy.txt"):
        assert (output / name).read_bytes() == (TEST_DATA / name).read_bytes()
    assert json.loads((output / "benchmark.json").read_text()) == report
    assert (output / "benchmark.md").read_text().startswith("# Policy quality benchmark")
    assert json.loads((output / "human-review.json").read_text())["status"] == "pending_human_review"
    for row in report["results"]:
        assert json.loads((output / row["artifact_directory"] / "result.json").read_text()) == row


@title("A run creates a nested output directory and stamps its manifest in UTC")
def test_run_creates_nested_output(benchmark_run):
    benchmark_run.output = benchmark_run.root.parent / "runs" / "2026-10-10"

    benchmark_run.run()

    created = json.loads((benchmark_run.output / "manifest.json").read_text())["created_at"]
    assert datetime.fromisoformat(created).utcoffset() == timedelta(0)


@title("A run never reuses an existing output directory")
def test_run_refuses_existing_output(benchmark_run):
    benchmark_run.output.mkdir()

    with pytest.raises(FileExistsError):
        benchmark_run.run()

    benchmark_run.generate.assert_not_called()


@pytest.mark.parametrize(
        "catalog", [
        pytest.param(Mock(side_effect=RuntimeError("Offline")), id="unreachable"),
        pytest.param(lambda _: model_catalog(("other-model", JUDGE_DIGEST)), id="model-not-installed"),
        pytest.param(lambda _: model_catalog(status_code=503), id="catalog-error")])
@title("A failed model preflight records every planned row as an error without any generation [{param_id}]")
def test_run_preflight_failure(benchmark_run, monkeypatch, catalog):
    monkeypatch.setattr(runner.OllamaClient, "list_models", catalog)

    report = benchmark_run.run()

    assert (report["status"], report["summary"]["errors"]) == ("error", 2)
    assert all(row["error"] == "ValueError: Model preflight did not complete" for row in report["results"])
    assert report["calibration"]["error"].startswith("Model preflight failed: ")
    benchmark_run.generate.assert_not_called()


@title("A generation whose weights differ from the preflight digest is an error for that case")
def test_run_rejects_changed_weights(benchmark_run):
    benchmark_run.catalog = model_catalog((MODEL, "other-digest"))

    report = benchmark_run.run()

    assert all(row["error"] == "ValueError: Generation weights changed after preflight" for row in report["results"])
    assert report["calibration"]["error"] == "No paid-leave sample was available for judge controls"
    benchmark_run.calibrate.assert_not_called()


@title("A failed generation is recorded for its case and the remaining cases still run")
def test_run_continues_after_failed_generation(benchmark_run):
    original = benchmark_run.generate.side_effect

    def fail_refusal(root, directory, case_id, model):
        if case_id == "gym_missing":
            raise RuntimeError("AnythingLLM unavailable")
        return original(root, directory, case_id, model)

    benchmark_run.generate.side_effect = fail_refusal

    report = benchmark_run.run()

    assert [row.get("error") for row in report["results"]] == [None, "RuntimeError: AnythingLLM unavailable"]
    assert report["status"] == "error"


@title("Evidence that summarisation rejects is saved as an aggregation error without a Markdown summary")
def test_run_saves_aggregation_error(benchmark_run, monkeypatch):
    monkeypatch.setattr(runner, "summarize", Mock(side_effect=ValueError("Unknown or duplicate benchmark result")))

    report = benchmark_run.run()

    assert report["status"] == "error"
    assert report["error"] == (
            "Aggregation rejected inconsistent evidence: ValueError: "
            "Unknown or duplicate benchmark result")
    assert (report["schema_version"], len(report["results"])) == (1, 2)
    assert report["manifest"]["expected_rows"] == make_definition(("gym_missing", "paid_leave"))["expected_rows"]
    assert not (benchmark_run.output / "benchmark.md").exists()
    assert not (benchmark_run.output / "human-review.json").exists()


@pytest.fixture
def cli(automation_root, monkeypatch):
    """The benchmark CLI with the run replaced, so that options, checks and exit codes are verified offline."""
    state = SimpleNamespace(root=automation_root, output=automation_root.parent / "new")
    state.run = Mock(return_value={"status": "checks_passed"})
    monkeypatch.setattr(runner, "run", state.run)

    def invoke(*arguments: str) -> int:
        monkeypatch.setattr(
                "sys.argv", ["benchmark", "--root",
                str(automation_root), "--output",
                str(state.output), *arguments])
        return runner.main()

    state.invoke = invoke
    return state


@title("A dry run prints the manifest without running, connecting or creating the output directory")
def test_cli_dry_run(cli, capsys):
    assert cli.invoke("--dry-run") == 0

    printed = json.loads(capsys.readouterr().out)
    assert (printed["maximum_model_calls"], printed["control_ids"]) == (43, list(CONTROL_IDS))
    cli.run.assert_not_called()
    assert not cli.output.exists()


@title("The CLI plans the selected cases, models, judge and budget and reports a passing benchmark with 0")
def test_cli_runs_selected_plan(cli, capsys):
    exit_code = cli.invoke(
            "--case", "paid_leave", "--case", "gym_missing", "--model", MODEL, "--model", "qwen2.5:7b", "--judge-model",
            MODEL, "--max-model-calls", "40")

    root, output, plan, dataset, gates = cli.run.call_args.args
    assert exit_code == 0
    assert (root, output) == (cli.root.resolve(), cli.output.resolve())
    assert (plan.case_ids, plan.models, plan.maximum_calls) == (("paid_leave", "gym_missing"), (MODEL, "qwen2.5:7b"),
            6 + 2 * 13)
    assert (dataset.sha256, gates) == (GOLDEN_DATASET.sha256, QUALITY_GATES)
    printed = capsys.readouterr().out
    assert "Serial benchmark: 4 generations; at most 32 generation/judge calls; no retries." in printed
    assert f"Benchmark: checks_passed; evidence: {cli.output.resolve()}" in printed


@title("A benchmark that did not pass exits with 1")
def test_cli_reports_failed_benchmark(cli):
    cli.run.return_value = {"status": "failed"}

    assert cli.invoke() == 1


@pytest.mark.parametrize(("prepare", "arguments", "message"), [
        pytest.param(lambda state: state.output.mkdir(), [], "Output already exists", id="existing-output"),
        pytest.param(
        lambda state: (state.root / "tests/test_golden_rag.py").unlink(), [], "Run from automation or provide --root",
        id="not-an-automation-root"),
        pytest.param(lambda state: None, ["--case", "unknown"], "unique, known golden cases", id="invalid-plan"),
        pytest.param(lambda state: None, ["--max-model-calls", "10"], "needs at most 43 model calls", id="over-budget")
])
@title("The CLI refuses an unusable output, root or plan before any model call [{param_id}]")
def test_cli_refuses_unusable_invocation(cli, capsys, prepare, arguments, message):
    prepare(cli)

    with pytest.raises(SystemExit) as stopped:
        cli.invoke(*arguments)

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err
    cli.run.assert_not_called()


@title("The CLI refuses a real run without the evaluation dependencies, but still allows a dry run")
def test_cli_requires_evaluation_dependencies(cli, monkeypatch, capsys):
    monkeypatch.setattr(runner, "importlib", Mock(util=Mock(find_spec=Mock(return_value=None))))

    assert cli.invoke("--dry-run") == 0
    capsys.readouterr()
    with pytest.raises(SystemExit) as stopped:
        cli.invoke()

    assert stopped.value.code == 2
    assert "Install requirements-evaluation.lock" in capsys.readouterr().err


@title("The answer request duration is measured around the chat call and recorded in JUnit metadata")
def test_chat_duration_is_measured(monkeypatch, tmp_path):
    # Imported here: a module-level import would register rag_chat as a fixture of this module.
    from test_support.fixtures.rag import rag_chat

    clock = Mock(side_effect=(100.0, 112.5))
    monkeypatch.setattr("test_support.fixtures.rag.perf_counter", clock)
    client, record_property = Mock(), Mock()
    chat = rag_chat.__wrapped__(
            request=Mock(), rag_environment=None, automation_root=tmp_path, authenticated_anythingllm_api=client,
            indexed_workspace={"slug": "temporary"}, settings=Settings(), capture_id=None, generation_model=MODEL,
            rag_iteration=1, generation_model_digest="digest", workspace_configuration={},
            policy_file=tmp_path / "unused.txt", record_property=record_property)

    result = chat("Leave?", "Reference")

    assert result is client.chat.return_value
    record_property.assert_called_once_with("answer_request_seconds", 12.5)
    assert clock.call_count == 2
