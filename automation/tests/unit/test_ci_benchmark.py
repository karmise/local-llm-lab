"""CI benchmark: reviewed presets and saved evidence bound to the tested revision, checkout and model lock."""

import json
import sys
from unittest.mock import Mock

import pytest

from llm_testkit.ci import benchmark as ci
from llm_testkit.ci.benchmark import BASELINES, execution_context, inputs
from llm_testkit.datasets.benchmark import manifest
from llm_testkit.qualification.plan import framework_checksum
from llm_testkit.reporting.steps import title
from test_support.builders.benchmark import JUDGE_DIGEST, MODEL
from test_support.builders.ci import COMPARISON_MODEL, OTHER_REVISION, REVISION, ci_metadata, save_ci_run
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("profile", "models", "cases", "planned_models", "maximum_calls"), [
        pytest.param("smoke", "primary", 2, [MODEL], 19, id="smoke-primary"),
        pytest.param("curated", "primary", 4, [MODEL], 43, id="curated-primary"),
        pytest.param("smoke", "comparison", 2, [MODEL, COMPARISON_MODEL], 32, id="smoke-comparison"),
        pytest.param("curated", "comparison", 4, [MODEL, COMPARISON_MODEL], 80, id="curated-comparison")])
@title("CI presets declare a bounded, complete matrix before any model call [{param_id}]")
def test_ci_presets(profile, models, cases, planned_models, maximum_calls):
    dataset, gates, plan = inputs(AUTOMATION_ROOT, profile, models)

    assert (len(plan.case_ids), list(plan.models), plan.maximum_calls) == (cases, planned_models, maximum_calls)
    assert set(plan.case_ids) <= {case.id for case in dataset.cases}
    assert gates["calibration"] == "experimental"


@title("The smoke profile runs paid leave and the missing-information refusal")
def test_smoke_profile_cases():
    _, _, plan = inputs(AUTOMATION_ROOT, "smoke", "primary")

    assert list(plan.case_ids) == ["paid_leave", "gym_missing"]


@pytest.mark.parametrize(("profile", "models"),
        [pytest.param("full", "primary", id="unknown-profile"),
        pytest.param("smoke", "all", id="unknown-models")])
@title("Only reviewed profiles and model presets can be selected [{param_id}]")
def test_inputs_reject_unreviewed_selection(profile, models):
    with pytest.raises(ValueError, match="Select a reviewed CI profile and model preset"):
        inputs(AUTOMATION_ROOT, profile, models)


@title("The execution context names the full revision, scope and framework source checksum")
def test_execution_context():
    context = execution_context(AUTOMATION_ROOT, REVISION, "smoke", "primary")

    assert context == {
            "schema_version": 1,
            "revision": REVISION,
            "profile": "smoke",
            "models": "primary",
            "framework_source_sha256": framework_checksum(AUTOMATION_ROOT)}


@pytest.mark.parametrize(
        "revision", [
        pytest.param(REVISION[:7], id="abbreviated"),
        pytest.param("A" * 40, id="uppercase"),
        pytest.param(REVISION + "a", id="too-long"),
        pytest.param("g" * 40, id="not-hex")])
@title("CI evidence requires the full lowercase commit SHA [{param_id}]")
def test_execution_context_rejects_partial_revision(revision):
    with pytest.raises(ValueError, match="CI evidence requires the full tested commit SHA"):
        execution_context(AUTOMATION_ROOT, revision, "smoke", "primary")


@pytest.fixture
def ci_run(tmp_path, monkeypatch):
    return save_ci_run(tmp_path, monkeypatch)


@title("Fresh CI evidence passes; the manifest creation time is not part of the compared plan")
def test_saved_ci_run_passes(ci_run):
    report = ci_run.validate()

    assert report["status"] == "checks_passed"
    assert [row["case_id"] for row in report["results"]] == ["paid_leave", "gym_missing"]
    assert "created_at" in ci_run.read("manifest.json")


@pytest.mark.parametrize(("revision", "profile", "models"), [
        pytest.param(OTHER_REVISION, "smoke", "primary", id="other-revision"),
        pytest.param(REVISION, "curated", "primary", id="other-profile"),
        pytest.param(REVISION, "smoke", "comparison", id="other-models")])
@title("Evidence is rejected for any other revision or requested scope [{param_id}]")
def test_ci_run_rejects_other_scope(ci_run, revision, profile, models):
    with pytest.raises(ValueError, match="CI context differs from the tested revision, source or requested scope"):
        ci_run.validate(revision=revision, profile=profile, models=models)


@title("Evidence is rejected when the framework source changed after the run")
def test_ci_run_rejects_changed_framework(ci_run):
    (ci_run.root / "pyproject.toml").write_text("[project]\nname = 'changed'\n")

    with pytest.raises(ValueError, match="CI context differs"):
        ci_run.validate()


@pytest.mark.parametrize("filename", BASELINES)
@title("Every reviewed baseline saved with the run must match the checkout [{filename}]")
def test_ci_run_rejects_changed_baseline(ci_run, filename):
    with (ci_run.directory / filename).open("a") as file:
        file.write("\n")

    with pytest.raises(ValueError, match=f"^CI baseline differs from the tested checkout: {filename}$"):
        ci_run.validate()


@title("A saved manifest that omits a planned row is rejected")
def test_ci_run_rejects_incomplete_manifest(ci_run):
    definition = ci_run.read("manifest.json")
    definition["expected_rows"].pop()
    ci_run.write("manifest.json", definition)

    with pytest.raises(ValueError, match="CI manifest differs from the complete requested plan"):
        ci_run.validate()


@title("Judge weights that differ from the reviewed model lock are rejected")
def test_ci_run_rejects_other_judge_weights(ci_run):
    ci_run.lock.write_text(json.dumps({MODEL: "changed-weights", COMPARISON_MODEL: JUDGE_DIGEST}))

    with pytest.raises(ValueError, match="CI judge weights differ from the reviewed model lock"):
        ci_run.validate()


@title("Generation weights that differ from the reviewed model lock are rejected")
def test_ci_run_rejects_other_generation_weights(tmp_path, monkeypatch):
    run = save_ci_run(tmp_path, monkeypatch, models="comparison")
    run.lock.write_text(json.dumps({MODEL: JUDGE_DIGEST, COMPARISON_MODEL: "changed-weights"}))

    with pytest.raises(ValueError, match="CI generation weights differ from the reviewed model lock"):
        run.validate()


@title("Comparison-model evidence with locked weights passes")
def test_comparison_ci_run_passes(tmp_path, monkeypatch):
    run = save_ci_run(tmp_path, monkeypatch, models="comparison")

    assert {row["model"] for row in run.validate()["results"]} == {MODEL, COMPARISON_MODEL}


@pytest.mark.parametrize("field", ["framework_source_sha256", "test_source_sha256", "test_node_id"])
@title("A sample not produced by the tested framework and golden test is rejected [{field}]")
def test_ci_run_rejects_foreign_sample(tmp_path, monkeypatch, field):
    def foreign(root, case_id, model):
        return ci_metadata(root, case_id, model) | {field: "other"}

    run = save_ci_run(tmp_path, monkeypatch, metadata=foreign)

    with pytest.raises(ValueError, match="CI sample was not produced by the tested framework and golden test"):
        run.validate()


@title("A golden test changed after the run makes its samples foreign")
def test_ci_run_rejects_changed_golden_test(ci_run):
    (ci_run.root / "tests/test_golden_rag.py").write_text("# Different golden test\n")

    with pytest.raises(ValueError, match="CI sample was not produced by the tested framework and golden test"):
        ci_run.validate()


@title("A generation error keeps the CI verdict failing even if a saved summary claimed success")
def test_ci_run_keeps_generation_error(ci_run):
    report = ci_run.read("benchmark.json")
    row = report["results"][0]
    row["error"] = "Generation could not complete"
    ci_run.write(f"{row['artifact_directory']}/result.json", row)
    ci_run.write("benchmark.json", report)

    assert ci_run.validate()["status"] == "error"


@title("The sample of an errored row is not required (the refusal row; paid leave binds the judge controls)")
def test_ci_run_skips_errored_row_sample(ci_run):
    report = ci_run.read("benchmark.json")
    row = report["results"][1]
    row["error"] = "Generation could not complete"
    ci_run.write(f"{row['artifact_directory']}/result.json", row)
    ci_run.write("benchmark.json", report)
    (ci_run.directory / row["artifact_directory"] / "sample.json").unlink()

    assert ci_run.validate()["status"] == "error"


def foreign_refusal(root, case_id, model):
    """CI metadata in which only the refusal case's sample names another test node."""
    return ci_metadata(root, case_id, model) | ({"test_node_id": "other"} if case_id == "gym_missing" else {})


@title("Rows after an errored row are still verified")
def test_ci_run_verifies_rows_after_error(tmp_path, monkeypatch):
    run = save_ci_run(tmp_path, monkeypatch, metadata=foreign_refusal)
    report = run.read("benchmark.json")
    row = report["results"][0]
    row["error"] = "Generation could not complete"
    run.write(f"{row['artifact_directory']}/result.json", row)
    run.write("benchmark.json", report)

    with pytest.raises(ValueError, match="CI sample was not produced by the tested framework"):
        run.validate()


@title("A manifest without its creation time passes the CI plan comparison and is checked against its artifact")
def test_ci_run_tolerates_manifest_without_creation_time(ci_run):
    definition = ci_run.read("manifest.json")
    del definition["created_at"]
    ci_run.write("manifest.json", definition)

    with pytest.raises(ValueError, match="Saved benchmark manifest differs from its artifact"):
        ci_run.validate()


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["ci-benchmark", *map(str, args)])
    return ci.main()


@title("The plan operation writes the context and complete manifest without running anything")
def test_cli_plan(tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr(ci, "run", run)

    code = run_cli(
            monkeypatch, "plan", "--root", AUTOMATION_ROOT, "--profile", "smoke", "--revision", REVISION, "--output",
            tmp_path / "plan.json")

    planned = json.loads((tmp_path / "plan.json").read_text())
    assert code == 0
    assert planned["context"] == execution_context(AUTOMATION_ROOT, REVISION, "smoke", "primary")
    dataset, gates, plan = inputs(AUTOMATION_ROOT, "smoke", "primary")
    expected = manifest(plan, dataset, gates, AUTOMATION_ROOT / "test_data/faithfulness-controls.json")
    assert {
            k: v
            for k, v in planned["manifest"].items() if k != "created_at"} == {
            k: v
            for k, v in expected.items() if k != "created_at"}
    run.assert_not_called()


@title("The plan operation defaults to the curated profile on the primary model")
def test_cli_plan_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(AUTOMATION_ROOT)

    run_cli(monkeypatch, "plan", "--revision", REVISION, "--output", tmp_path / "plan.json")

    context = json.loads((tmp_path / "plan.json").read_text())["context"]
    assert (context["profile"], context["models"]) == ("curated", "primary")


@pytest.mark.parametrize(("status", "code"), [
        pytest.param("checks_passed", 0, id="passed"),
        pytest.param("failed", 1, id="failed"),
        pytest.param("error", 1, id="error")])
@title("The produce operation runs the plan, records its context and exits non-zero unless checks passed [{param_id}]")
def test_cli_produce(tmp_path, monkeypatch, capsys, status, code):
    run = Mock(return_value={"status": status})
    monkeypatch.setattr(ci, "run", run)
    output = tmp_path / "run"
    output.mkdir()

    exit_code = run_cli(
            monkeypatch, "produce", "--root", AUTOMATION_ROOT, "--profile", "smoke", "--models", "comparison",
            "--revision", REVISION, "--output", output)

    root, directory, plan, dataset, gates = run.call_args.args
    expected_dataset, expected_gates, _ = inputs(AUTOMATION_ROOT, "smoke", "comparison")
    assert exit_code == code
    assert (root, directory,
            list(plan.models)) == (AUTOMATION_ROOT.resolve(), output.resolve(), [MODEL, COMPARISON_MODEL])
    assert (dataset.sha256, gates) == (expected_dataset.sha256, expected_gates)
    assert json.loads((output / "ci-context.json").read_text())["models"] == "comparison"
    assert capsys.readouterr().out == f"CI benchmark: {status}; maximum generation/judge calls: 32\n"


@title("A partial revision is refused before anything is planned or run")
def test_cli_rejects_partial_revision(tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr(ci, "run", run)

    with pytest.raises(ValueError, match="full tested commit SHA"):
        run_cli(monkeypatch, "produce", "--root", AUTOMATION_ROOT, "--revision", "abc1234", "--output", tmp_path)
    run.assert_not_called()


@pytest.mark.parametrize(
        "arguments", [
        pytest.param(["plan", "--output", "out.json"], id="no-revision"),
        pytest.param(["plan", "--revision", REVISION], id="no-output"),
        pytest.param(["publish", "--revision", REVISION, "--output", "out.json"], id="unknown-operation"),
        pytest.param(["plan", "--profile", "full", "--revision", REVISION, "--output", "out.json"],
        id="unknown-profile"),
        pytest.param(["plan", "--models", "all", "--revision", REVISION, "--output", "out.json"], id="unknown-models")])
@title("The command requires a known operation, a revision and an output [{param_id}]")
def test_cli_rejects_incomplete_arguments(tmp_path, monkeypatch, arguments):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exit:
        run_cli(monkeypatch, *arguments)

    assert exit.value.code == 2
    assert list(tmp_path.iterdir()) == []
