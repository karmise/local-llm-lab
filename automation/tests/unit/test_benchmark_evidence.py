"""Offline revalidation of a saved benchmark: every summary must be reproducible from its original artifacts.

Each tampering case edits all copies of a value that the earlier checks compare, so that only the named rule can fire.
"""

import hashlib
import json
import shutil
from unittest.mock import Mock

import pytest

from llm_testkit.datasets.benchmark import make_plan
from llm_testkit.evaluation import benchmark_runner as runner
from llm_testkit.reporting.benchmark_evidence import load_saved_benchmark
from llm_testkit.reporting.steps import title
from test_support.builders.benchmark import (
        JUDGE_DIGEST, MODEL, QUALITY_GATES, make_calibrate_stub, make_calibration, make_definition, make_generate_stub)
from test_support.builders.golden import GOLDEN_DATASET, TEST_DATA
from test_support.builders.ollama import model_catalog

pytestmark = pytest.mark.unit


def run_benchmark(tmp_path, monkeypatch, *, catalog=None):
    root = tmp_path / "automation"
    shutil.copytree(TEST_DATA, root / "test_data")
    output = tmp_path / "run"
    monkeypatch.delenv("ANYTHINGLLM_API_KEY", raising=False)
    monkeypatch.setattr(runner.OllamaClient, "list_models", catalog or (lambda _: model_catalog((MODEL, JUDGE_DIGEST))))
    monkeypatch.setattr(runner, "calibrate", make_calibrate_stub(make_calibration(make_definition())))
    monkeypatch.setattr(runner, "generate_sample", make_generate_stub(GOLDEN_DATASET, monkeypatch))
    plan = make_plan(GOLDEN_DATASET, case_ids=["paid_leave", "gym_missing"])
    runner.run(root, output, plan, GOLDEN_DATASET, QUALITY_GATES, notify=Mock())
    return output


@pytest.fixture
def saved(tmp_path, monkeypatch):
    """A complete passing benchmark run saved to disk: case-001 is paid leave, case-002 the refusal."""
    return run_benchmark(tmp_path, monkeypatch)


def edit_json(path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def edit_report(output, change) -> None:
    edit_json(output / "benchmark.json", change)


def edit_row(output, index: int, change) -> None:
    """Change a result consistently in the saved report and in its case artifact."""
    edit_report(output, lambda report: change(report["results"][index]))
    edit_json(output / f"case-00{index + 1}/result.json", change)


def edit_calibration(output, change) -> None:
    """Change the judge-control summary consistently in the saved report and in its artifact."""
    edit_report(output, lambda report: change(report["calibration"]))
    edit_json(output / "judge-controls.json", change)


def edit_manifest(output, change) -> None:
    edit_report(output, lambda report: change(report["manifest"]))
    edit_json(output / "manifest.json", change)


def replace_sample(output, index: int, change) -> None:
    """Edit a case sample and re-bind its row to the new sample checksum."""
    sample_path = output / f"case-00{index + 1}/sample.json"
    edit_json(sample_path, change)
    checksum = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    edit_row(output, index, lambda row: row.update(sample_sha256=checksum))


def append(path, text: str) -> None:
    path.write_text(path.read_text() + text)


@title("A saved run is reloaded by recomputing its summary from the original artifacts")
def test_saved_run_is_reproducible(saved):
    original = json.loads((saved / "benchmark.json").read_text())

    reloaded = load_saved_benchmark(saved / "benchmark.json")

    assert reloaded["status"] == "checks_passed"
    assert reloaded["summary"] == original["summary"]
    assert reloaded["results"] == original["results"]


@title("A forged top-level status and summary are ignored: both are recomputed from the recorded rows")
def test_forged_status_is_recomputed(saved):
    edit_report(saved, lambda report: report.update(status="failed", summary={"passed": 999}))

    reloaded = load_saved_benchmark(saved / "benchmark.json")

    assert reloaded["status"] == "checks_passed"
    assert (reloaded["summary"]["planned"], reloaded["summary"]["passed"]) == (2, 2)


@title("A saved report without results or matched controls is reloaded as an error with every row missing")
def test_empty_saved_report_is_an_error(saved):
    edit_report(saved, lambda report: report.update(results=[], calibration={"status": "error"}))

    reloaded = load_saved_benchmark(saved / "benchmark.json")

    assert (reloaded["status"], reloaded["summary"]["missing"]) == ("error", 2)


@title("A run whose preflight failed is reloaded without generation artifacts and stays an error")
def test_failed_run_is_reloaded(tmp_path, monkeypatch):
    output = run_benchmark(tmp_path, monkeypatch, catalog=Mock(side_effect=RuntimeError("Offline")))

    reloaded = load_saved_benchmark(output / "benchmark.json")

    assert (reloaded["status"], reloaded["summary"]["errors"]) == ("error", 2)


def record_timing_missing_from_junit(output) -> None:
    """A duration consistent between sample and row, but absent from the generation JUnit."""
    replace_sample(output, 1, lambda sample: sample["metadata"].update(answer_request_seconds=2.0))
    edit_row(output, 1, lambda row: row.update(answer_request_seconds=2.0))


def flip_first_control_statement(calibration: dict) -> None:
    calibration["results"][0]["judge_calls"][0]["output"]["statements"] = ["Edited statement"]


TAMPERING = [
        pytest.param(
        lambda out: edit_json(out / "manifest.json", lambda m: m.update(judge_model="other")),
        "manifest differs from its artifact", id="manifest"),
        pytest.param(
        lambda out: append(out / "golden-policy.json", "\n"), "dataset checksum mismatch", id="golden-dataset"),
        pytest.param(
        lambda out: append(out / "company-policy.txt", "\n"), "Golden dataset policy checksum mismatch",
        id="policy-file"),
        pytest.param(
        lambda out: edit_manifest(out, lambda m: m.update(policy_sha256="other")),
        "Benchmark dataset checksum mismatch", id="manifest-policy-checksum"),
        pytest.param(
        lambda out: edit_manifest(out, lambda m: m.update(golden_dataset_version="other")), "dataset checksum mismatch",
        id="dataset-version"),
        pytest.param(
        lambda out: edit_json(out / "quality-gates.json", lambda g: g["minimum_scores"].update(faithfulness=0.5)),
        "gates differ from their recorded baseline", id="quality-gates"),
        pytest.param(
        lambda out: edit_manifest(out, lambda m: m["expected_rows"][0].update(category="boundary")),
        "matrix differs from its golden cases", id="matrix-category"),
        pytest.param(
        lambda out: edit_manifest(out, lambda m: m["expected_rows"][0].update(case_id="unknown")),
        "matrix differs from its golden cases", id="matrix-case"),
        pytest.param(
        lambda out: edit_json(out / "judge-controls.json", lambda c: c.update(status="mismatch")),
        "Judge control summary differs from its artifact", id="judge-control-summary"),
        pytest.param(
        lambda out: edit_calibration(out, lambda c: c.update(sample_sha256="other")),
        "not bound to a paid-leave benchmark sample", id="controls-unanchored"),
        pytest.param(
        lambda out: append(out / "faithfulness-controls.json", "\n"), "control catalog checksum mismatch",
        id="control-catalog"),
        pytest.param(
        lambda out: edit_calibration(out, lambda c: c["results"][0]["control"].update(id="other")),
        "labels differ from their catalog", id="control-labels"),
        pytest.param(
        lambda out: edit_calibration(out, flip_first_control_statement), "verdicts differ from their raw calls",
        id="control-raw-calls"),
        pytest.param(
        lambda out: edit_calibration(out, lambda c: c["results"][0]["judge_calls"].pop()),
        "verdicts differ from their raw calls", id="control-call-missing"),
        pytest.param(
        lambda out: edit_report(out, lambda r: r["results"][1].update(artifact_directory="../case-002")),
        "Invalid benchmark artifact directory", id="artifact-directory"),
        pytest.param(
        lambda out: edit_json(out / "case-002/result.json", lambda r: r.update(answer="Edited")),
        "differs from its original case artifact", id="case-artifact"),
        pytest.param(
        lambda out: append(out / "case-002/generation.xml", "\n"), "JUnit checksum/size mismatch",
        id="generation-junit"),
        pytest.param(
        lambda out: edit_row(out, 1, lambda r: r.update(generation_status="failed")),
        "generation status differs from JUnit", id="generation-status"),
        pytest.param(
        lambda out: append(out / "case-002/sample.json", "\n"), "sample/answer checksum mismatch", id="sample"),
        pytest.param(
        lambda out: edit_row(out, 1, lambda r: r.update(answer="Edited")), "sample/answer checksum mismatch",
        id="answer"),
        pytest.param(
        lambda out: edit_row(out, 1, lambda r: r.update(answer_request_seconds=1.0)),
        "timing differs from captured sample metadata", id="timing-without-measurement"),
        pytest.param(
        record_timing_missing_from_junit, "timing differs from generation JUnit", id="timing-not-in-junit"),
        pytest.param(
        lambda out: edit_row(out, 1, lambda r: r["dimensions"][1].update(status="failed")),
        "answer/source checks differ from actual inputs", id="acceptance-checks"),
        pytest.param(
        lambda out: append(out / "case-001/faithfulness.json", "\n"), "metric evidence checksum mismatch",
        id="metric-evidence"),
        pytest.param(
        lambda out: edit_row(out, 0, lambda r: r["dimensions"][5].update(value=0.95)),
        "metric summary differs from original judge evidence", id="metric-summary")]


@pytest.mark.parametrize(("tamper", "message"), TAMPERING)
@title("Tampering with any saved artifact or its summary is detected by the matching integrity rule [{param_id}]")
def test_tampering_is_detected(saved, tamper, message):
    tamper(saved)

    with pytest.raises(ValueError, match=message):
        load_saved_benchmark(saved / "benchmark.json")


def bind_junit_timing(output, index: int, seconds: float) -> None:
    """Record one duration consistently in the sample, the generation JUnit and the row."""
    directory = output / f"case-00{index + 1}"
    replace_sample(output, index, lambda sample: sample["metadata"].update(answer_request_seconds=seconds))
    junit = directory / "generation.xml"
    junit.write_text(
            junit.read_text().replace(
            "/>", f'><properties><property name="answer_request_seconds" value="{seconds}"/></properties></testcase>',
            1))
    checksum = hashlib.sha256(junit.read_bytes()).hexdigest()
    edit_row(output, index, lambda row: row.update(answer_request_seconds=seconds, generation_junit_sha256=checksum))


@title("A duration recorded consistently in the sample, JUnit and row is accepted")
def test_consistent_timing_is_accepted(saved):
    bind_junit_timing(saved, 1, 2.0)

    reloaded = load_saved_benchmark(saved / "benchmark.json")

    assert reloaded["summary"]["answer_timing"]["measured"] == 1


@pytest.mark.parametrize(
        "earlier_row", [
        pytest.param(lambda row: None, id="refusal-before-answer"),
        pytest.param(lambda row: row.update(error="Generation crashed"), id="errored-row-before-answer")])
@title("Rows after a refusal or errored row are still checked against their artifacts [{param_id}]")
def test_rows_after_skipped_row_are_checked(saved, earlier_row):
    edit_row(saved, 1, earlier_row)
    edit_report(saved, lambda report: report["results"].reverse())
    append(saved / "case-001/faithfulness.json", "\n")

    with pytest.raises(ValueError, match="metric evidence checksum mismatch"):
        load_saved_benchmark(saved / "benchmark.json")


@title("A control summary with fewer results than declared controls is rejected")
def test_missing_control_result_is_rejected(saved):
    edit_calibration(saved, lambda calibration: calibration["results"].pop())

    with pytest.raises(ValueError, match="zip\\(\\) argument"):
        load_saved_benchmark(saved / "benchmark.json")


@title("Control results that contradict the reviewed labels are rejected even with consistent raw calls")
def test_control_contradicting_labels_is_rejected(saved):
    def contradict(calibration: dict) -> None:
        control = calibration["results"][0]
        for verdict in control["result"]["verdicts"]:
            verdict["verdict"] = 1 - verdict["verdict"]
        control["result"]["value"] = sum(v["verdict"]
                for v in control["result"]["verdicts"]) / len(control["result"]["verdicts"])
        control["judge_calls"][1]["output"]["statements"] = control["result"]["verdicts"]

    edit_calibration(saved, contradict)

    with pytest.raises(AssertionError, match="Control score"):
        load_saved_benchmark(saved / "benchmark.json")
