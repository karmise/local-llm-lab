"""Judge review: disagreement candidates from a verified saved benchmark, without approving or changing anything."""

import copy
import hashlib
import json

import pytest

from llm_testkit.reporting import judge_review
from llm_testkit.reporting.benchmark_evidence import load_saved_benchmark
from llm_testkit.reporting.judge_review import main, review_benchmark
from llm_testkit.reporting.steps import title
from test_support.builders.benchmark import JUDGE_DIGEST, MODEL, save_benchmark_run

pytestmark = pytest.mark.unit


@pytest.fixture
def disputed(tmp_path, monkeypatch):
    """A saved run whose paid-leave answer passes its checks while the correctness judge rejects it."""
    return save_benchmark_run(tmp_path, monkeypatch, dispute_correctness=True)


@title("A judge failure on an answer that passed its reviewed checks is a disagreement candidate with its evidence")
def test_review_reports_candidate_with_evidence(disputed):
    report = review_benchmark(disputed / "benchmark.json")

    candidate = report["rows"][0]
    saved_row = json.loads((disputed / "case-001/result.json").read_text())
    raw = (disputed / "case-001/correctness.json").read_bytes()
    assert candidate["assessment"] == "judge_reference_disagreement_candidate"
    assert [d["metric"] for d in candidate["failed_semantic_dimensions"]] == ["factual_correctness"]
    assert candidate["evidence"] == {
            "correctness": {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "result": json.loads(raw)["result"],
            "error": None,
            "judge_model": MODEL,
            "judge_model_digest": JUDGE_DIGEST}}
    assert candidate["deterministic_checks"] == saved_row["dimensions"][:2]
    assert (candidate["case_id"], candidate["model"],
            candidate["sample_sha256"]) == ("paid_leave", MODEL, saved_row["sample_sha256"])
    assert (candidate["question"], candidate["reference"],
            candidate["answer"]) == (saved_row["question"], saved_row["reference"], saved_row["answer"])
    assert (candidate["generation_status"], candidate["human_verdict"]) == ("passed", None)


@title("The review keeps the benchmark status, records its inputs and leaves approval pending")
def test_review_does_not_approve(disputed):
    report = review_benchmark(disputed / "benchmark.json")

    manifest = json.loads((disputed / "manifest.json").read_text())
    assert report["schema_version"] == 1
    assert report["benchmark_sha256"] == hashlib.sha256((disputed / "benchmark.json").read_bytes()).hexdigest()
    assert (report["policy_sha256"],
            report["golden_dataset_sha256"]) == (manifest["policy_sha256"], manifest["golden_dataset_sha256"])
    assert report["original_benchmark_status"] == "checks_passed"
    assert (report["human_review"], report["threshold_decision"]) == ("pending", "retain_experimental_thresholds")
    assert "does not override failures, approve a judge or change any source artifact" in report["interpretation"]
    assert all(row["human_verdict"] is None for row in report["rows"])


@title("Without semantic failures no row is a candidate and no evidence is copied")
def test_review_of_passing_benchmark(tmp_path, monkeypatch):
    saved = save_benchmark_run(tmp_path, monkeypatch)

    report = review_benchmark(saved / "benchmark.json")

    assert report["original_benchmark_status"] == "checks_passed"
    assert [(row["assessment"], row["evidence"]) for row in report["rows"]] == [("no_automatic_assessment", {})] * 2


def loaded_with(disputed, change):
    """The verified benchmark with one row changed after verification, to isolate the assessment rule."""
    benchmark = load_saved_benchmark(disputed / "benchmark.json")
    changed = copy.deepcopy(benchmark)
    change(changed["results"][0])
    return changed


@pytest.mark.parametrize(
        "change", [
        pytest.param(lambda row: row["dimensions"][0].update(status="failed"), id="answer-check-failed"),
        pytest.param(lambda row: row.update(generation_status="error"), id="generation-not-passed"),
        pytest.param(lambda row: row.update(error="Teardown failed"), id="row-error")])
@title("A semantic failure is not a judge disagreement candidate unless the answer and its run were sound [{param_id}]")
def test_review_requires_sound_answer_for_candidate(disputed, monkeypatch, change):
    monkeypatch.setattr(judge_review, "load_saved_benchmark", lambda path: loaded_with(disputed, change))

    report = review_benchmark(disputed / "benchmark.json")

    assert report["rows"][0]["assessment"] == "no_automatic_assessment"
    assert set(report["rows"][0]["evidence"]) == {"correctness"}


@title("A metric error is also a candidate and copies the evidence of the failing evaluator only")
def test_review_includes_metric_errors(disputed, monkeypatch):
    def relevance_error(row):
        for dimension in row["dimensions"]:
            if dimension.get("metric") == "factual_correctness":
                dimension.update(status="passed", value=1.0)
            if dimension.get("metric") == "context_recall":
                dimension.update(status="error", error="Judge offline")

    monkeypatch.setattr(judge_review, "load_saved_benchmark", lambda path: loaded_with(disputed, relevance_error))

    row = review_benchmark(disputed / "benchmark.json")["rows"][0]

    assert row["assessment"] == "judge_reference_disagreement_candidate"
    assert set(row["evidence"]) == {"relevance"}
    assert row["evidence"]["relevance"]["error"] is None


@title("A row without dimensions is listed without assessment")
def test_review_lists_row_without_dimensions(disputed, monkeypatch):
    monkeypatch.setattr(
            judge_review, "load_saved_benchmark", lambda path: loaded_with(disputed, lambda row: row.pop("dimensions")))

    row = review_benchmark(disputed / "benchmark.json")["rows"][0]

    assert (row["assessment"], row["deterministic_checks"],
            row["failed_semantic_dimensions"]) == ("no_automatic_assessment", [], [])


@title("Corrupted benchmark evidence is rejected before any review row is produced")
def test_review_rejects_corrupted_evidence(disputed):
    (disputed / "case-001/faithfulness.json").write_text("{}")

    with pytest.raises(ValueError, match="checksum mismatch"):
        review_benchmark(disputed / "benchmark.json")


@title("The CLI saves the review, reports the number of candidates and exits with 0")
def test_cli_saves_review(disputed, tmp_path, monkeypatch, capsys):
    output = tmp_path / "review.json"
    monkeypatch.setattr("sys.argv", ["judge_review", str(disputed / "benchmark.json"), "--output", str(output)])

    assert main() == 0

    assert json.loads(output.read_text()) == review_benchmark(disputed / "benchmark.json")
    assert capsys.readouterr().out.strip() == (
            f"Verified benchmark: checks_passed; disagreement candidates: 1; human review pending. Saved to {output}")


@pytest.mark.parametrize(("prepare", "message"), [
        pytest.param(
        lambda disputed, output: output.write_text("existing"), "Output already exists", id="existing-output"),
        pytest.param(
        lambda disputed, output: (disputed / "case-001/faithfulness.json").write_text("{}"),
        "Cannot review invalid benchmark evidence: Benchmark metric evidence checksum mismatch", id="invalid-evidence")
])
@title("The CLI refuses to overwrite a review or to review invalid evidence [{param_id}]")
def test_cli_refuses(disputed, tmp_path, monkeypatch, capsys, prepare, message):
    output = tmp_path / "review.json"
    prepare(disputed, output)
    monkeypatch.setattr("sys.argv", ["judge_review", str(disputed / "benchmark.json"), "--output", str(output)])

    with pytest.raises(SystemExit) as stopped:
        main()

    assert stopped.value.code == 2
    assert message in capsys.readouterr().err


@title("The CLI requires an output file")
def test_cli_requires_output(disputed, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["judge_review", str(disputed / "benchmark.json")])

    with pytest.raises(SystemExit) as stopped:
        main()

    assert stopped.value.code == 2
