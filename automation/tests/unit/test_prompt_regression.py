"""Prompt regression: a versioned prompt catalog, controlled baseline/candidate comparison, the CLI and selection."""

import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

from llm_testkit.datasets.prompts import load_prompt_catalog
from llm_testkit.reporting import prompt_regression as reporting
from llm_testkit.reporting.prompt_regression import compare_prompts, main
from llm_testkit.reporting.steps import title
from test_support.builders.golden import GOLDEN_DATASET, GOLDEN_DATASET_FILE, POLICY_FILE, TEST_DATA
from test_support.builders.junit_xml import write_junit
from test_support.builders.prompt_regression import CLASSNAME, PROMPTS_FILE, catalog_json, prompt_catalog, prompt_run
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit


def load(data: dict, tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(data))
    return load_prompt_catalog(path)


@title("The catalog loads versioned variants with prompt checksums and names its baseline")
def test_catalog_loads_variants():
    raw, data = PROMPTS_FILE.read_bytes(), catalog_json()

    catalog = load_prompt_catalog(PROMPTS_FILE)

    assert (catalog.version, catalog.baseline,
            catalog.sha256) == (data["version"], "baseline", hashlib.sha256(raw).hexdigest())
    assert [(v.id, v.version) for v in catalog.variants] == [("baseline", "1"), ("grounded_v2", "2")]
    for variant, row in zip(catalog.variants, data["variants"], strict=True):
        assert (variant.prompt, variant.sha256) == (row["prompt"], hashlib.sha256(row["prompt"].encode()).hexdigest())


@title("The baseline prompt is the one the application workspace template uses")
def test_baseline_matches_workspace_template():
    template = json.loads((AUTOMATION_ROOT.parent / "config/workspace.json").read_text())

    assert prompt_catalog().variants[0].prompt == template["openAiPrompt"]


def second(**fields):
    return lambda data: data["variants"][1].update(fields)


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(lambda d: d.update(schema_version=2), "Unsupported prompt catalog schema", id="schema"),
        pytest.param(lambda d: d.pop("version"), "prompt catalog version must be", id="no-version"),
        pytest.param(lambda d: d["variants"].pop(), "at least two variants", id="one-variant"),
        pytest.param(lambda d: d.update(variants={}), "at least two variants", id="variants-not-a-list"),
        pytest.param(lambda d: d["variants"].__setitem__(1, "prompt"), "row must be an object", id="row-not-object"),
        pytest.param(second(id="Grounded"), "Invalid prompt id", id="invalid-id"),
        pytest.param(second(id=None), "Invalid prompt id", id="no-id"),
        pytest.param(second(id="baseline"), "Duplicate prompt id", id="duplicate-id"),
        pytest.param(second(prompt=""), "Prompt text and version must be nonempty", id="blank-prompt"),
        pytest.param(second(version=" "), "Prompt text and version must be nonempty", id="blank-version"),
        pytest.param(second(version=2), "Prompt text and version must be nonempty", id="numeric-version"),
        pytest.param(
        lambda d: d["variants"][1].update(prompt=d["variants"][1]["prompt"] + "[LLM_TESTKIT_CAPTURE:x]"),
        "must not contain runtime capture markers", id="capture-marker"),
        pytest.param(lambda d: d.update(baseline="absent"), "Baseline prompt is absent", id="unknown-baseline"),
        pytest.param(lambda d: d.pop("baseline"), "Baseline prompt is absent", id="no-baseline")])
@title("An invalid prompt catalog is rejected rule by rule [{param_id}]")
def test_catalog_rejects_invalid_catalog(tmp_path, corrupt, message):
    data = catalog_json()
    corrupt(data)

    with pytest.raises(ValueError, match=message):
        load(data, tmp_path)


@pytest.fixture
def comparison(tmp_path):
    """Compare JUnit evidence of baseline and grounded_v2 runs on the paid-leave case and one model."""
    baseline, candidate = prompt_catalog().variants

    def compare(*runs, cases=("paid_leave", ), models=("model", ), selected="grounded_v2", repeat=1):
        if not runs:
            runs = (prompt_run(baseline), prompt_run(candidate))
        path = write_junit(tmp_path / "runs.xml", list(runs), classname=CLASSNAME)
        report = compare_prompts(
                path, catalog_path=PROMPTS_FILE, dataset_path=GOLDEN_DATASET_FILE, policy_file=POLICY_FILE,
                case_ids=list(cases), models=list(models), candidate=selected, repeat=repeat)
        return report, path

    compare.baseline, compare.candidate = baseline, candidate
    return compare


@title("Two passing runs with identical settings pass, and the report records its scope and inputs")
def test_comparison_passes(comparison):
    report, path = comparison()

    assert (report["schema_version"], report["status"]) == (1, "passed")
    assert (report["baseline"], report["candidate"]) == ("baseline", "grounded_v2")
    assert report["catalog_sha256"] == prompt_catalog().sha256
    assert report["golden_dataset_sha256"] == GOLDEN_DATASET.sha256
    assert report["report_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["scope"] == {"cases": ["paid_leave"], "models": ["model"], "repeat": 1, "expected_runs": 2}
    assert (report["missing_runs"], report["errors"]) == ([], [])
    assert report["comparisons"] == [{
            "case": "paid_leave",
            "model": "model",
            "iteration": 1,
            "outcome": "passed",
            "baseline": "passed",
            "candidate": "passed"}]
    assert "not retries or statistical proof" in report["interpretation"]


@pytest.mark.parametrize(("baseline", "candidate", "status"), [
        pytest.param(None, "failure", "regression", id="candidate-failed"),
        pytest.param("failure", None, "baseline_failed", id="baseline-failed"),
        pytest.param("failure", "failure", "baseline_failed", id="both-failed"),
        pytest.param(None, "error", "incomplete", id="candidate-error"),
        pytest.param("error", None, "incomplete", id="baseline-error"),
        pytest.param("skipped", None, "incomplete", id="baseline-skipped"),
        pytest.param(None, "skipped", "incomplete", id="candidate-skipped")])
@title("Outcomes distinguish a regression, a failing baseline and incomplete evidence [{param_id}]")
def test_comparison_outcomes(comparison, baseline, candidate, status):
    report, _ = comparison(
            prompt_run(comparison.baseline, outcome=baseline), prompt_run(comparison.candidate, outcome=candidate))

    assert report["status"] == status
    assert report["comparisons"][0]["outcome"] == status


@title("A regression in one repetition outweighs a failing baseline in another")
def test_regression_takes_precedence(comparison):
    runs = [
            prompt_run(comparison.baseline, outcome="failure", name="b1"),
            prompt_run(comparison.candidate, name="c1"),
            prompt_run(comparison.baseline, rag_iteration="2", name="b2"),
            prompt_run(comparison.candidate, rag_iteration="2", outcome="failure", name="c2")]

    report, _ = comparison(*runs, repeat=2)

    assert [c["outcome"] for c in report["comparisons"]] == ["baseline_failed", "regression"]
    assert report["status"] == "regression"


@title("A missing run leaves the comparison incomplete and shows which side is missing")
def test_comparison_reports_missing_run(comparison):
    report, _ = comparison(prompt_run(comparison.baseline))

    assert report["status"] == "incomplete"
    assert report["missing_runs"] == [("paid_leave", "model", 1, "grounded_v2")]
    assert (report["comparisons"][0]["baseline"], report["comparisons"][0]["candidate"]) == ("passed", "missing")


@title("A missing baseline run is labelled on the baseline side")
def test_comparison_reports_missing_baseline(comparison):
    report, _ = comparison(prompt_run(comparison.candidate))

    assert report["missing_runs"] == [("paid_leave", "model", 1, "baseline")]
    assert (report["comparisons"][0]["baseline"], report["comparisons"][0]["candidate"]) == ("missing", "passed")


@title("Declared repetitions must all be present")
def test_comparison_requires_all_repetitions(comparison):
    report, _ = comparison(repeat=2)

    assert (report["status"], report["scope"]["expected_runs"], len(report["missing_runs"])) == ("incomplete", 4, 2)


@pytest.mark.parametrize(("changes", "error"), [
        pytest.param({"generation_model": "other"}, "Unexpected or duplicated matrix run", id="unplanned-model"),
        pytest.param({"golden_case_id": "carryover_limit"}, "Unexpected or duplicated", id="unplanned-case"),
        pytest.param({"rag_iteration": "x"}, "invalid literal", id="non-numeric-repetition"),
        pytest.param({"golden_dataset_sha256": "changed"}, "Changed or missing provenance", id="stale-dataset"),
        pytest.param({"policy_sha256": "changed"}, "Changed or missing provenance", id="stale-policy"),
        pytest.param({"prompt_catalog_sha256": "changed"}, "Changed or missing provenance", id="stale-catalog"),
        pytest.param({"prompt_sha256": "changed"}, "Changed or missing provenance", id="other-prompt-checksum"),
        pytest.param({"prompt_version": "9"}, "Changed or missing provenance", id="other-prompt-version"),
        pytest.param({"model_digest": ""}, "Changed or missing provenance", id="no-digest"),
        pytest.param({"thinking_mode": ""}, "Changed or missing provenance", id="no-thinking-mode"),
        pytest.param({"workspace_configuration": json.dumps({
        "chatModel": "model",
        "openAiPrompt": "Other"})}, "does not match selected prompt/model", id="other-prompt-text"),
        pytest.param({"prompt_id": None}, "'prompt_id'", id="no-prompt-id")])
@title("Duplicated, unplanned or uncontrolled runs make the comparison incomplete [{param_id}]")
def test_comparison_rejects_uncontrolled_runs(comparison, changes, error):
    report, _ = comparison(prompt_run(comparison.baseline), prompt_run(comparison.candidate, **changes))

    assert report["status"] == "incomplete"
    assert any(error in message for message in report["errors"])


@title("A second run of the same matrix cell is rejected as duplicated")
def test_comparison_rejects_duplicated_run(comparison):
    runs = prompt_run(comparison.baseline), prompt_run(comparison.candidate), prompt_run(
            comparison.candidate, name="again")

    report, _ = comparison(*runs)

    assert report["status"] == "incomplete"
    assert any("Unexpected or duplicated matrix run" in message for message in report["errors"])


@title("A candidate run on another chat model does not match the declared model")
def test_comparison_rejects_other_chat_model(comparison):
    other = json.dumps({"chatModel": "other", "openAiPrompt": comparison.candidate.prompt, "topN": 4})

    report, _ = comparison(
            prompt_run(comparison.baseline), prompt_run(comparison.candidate, workspace_configuration=other))

    assert any("does not match selected prompt/model" in message for message in report["errors"])


@title("Runs generated with different weights or retrieval settings are not compared")
def test_comparison_rejects_changed_setup(comparison):
    changed = json.dumps({"chatModel": "model", "openAiPrompt": comparison.candidate.prompt, "topN": 8})

    report, _ = comparison(
            prompt_run(comparison.baseline), prompt_run(comparison.candidate, workspace_configuration=changed))

    assert report["errors"] == ["paid_leave/model/1: model or retrieval configuration changed"]
    assert report["comparisons"][0]["outcome"] == "incomplete"


@title("The per-request capture marker in the prompt is not a prompt change")
def test_comparison_ignores_capture_marker(comparison):
    marked = json.dumps({
            "chatModel": "model",
            "openAiPrompt": comparison.candidate.prompt + "\n[LLM_TESTKIT_CAPTURE:" + "a" * 32 + "]",
            "topN": 4})

    report, _ = comparison(
            prompt_run(comparison.baseline), prompt_run(comparison.candidate, workspace_configuration=marked))

    assert report["status"] == "passed"


@pytest.mark.parametrize("conflict_first", [True, False], ids=["conflict-first", "conflict-last"])
@title("Conflicting metadata within one run is reported whatever its position [{param_id}]")
def test_comparison_rejects_conflicting_metadata(comparison, tmp_path, conflict_first):
    _, path = comparison()
    stale = '<property name="policy_sha256" value="stale"/>'
    xml = path.read_text()
    xml = xml.replace("<properties>", "<properties>" +
            stale, 1) if conflict_first else xml.replace("</properties>", stale + "</properties>", 1)
    path.write_text(xml)

    report = compare_prompts(
            path, catalog_path=PROMPTS_FILE, dataset_path=GOLDEN_DATASET_FILE, policy_file=POLICY_FILE,
            case_ids=["paid_leave"], models=["model"], candidate="grounded_v2")

    assert report["status"] == "incomplete"
    assert report["errors"] == [
            "('tests.test_prompt_regression', 'test_prompt[baseline]'): Conflicting metadata: ['policy_sha256']"]


@title("A teardown error is not hidden by a later failed call phase of the same test")
def test_teardown_error_is_retained(comparison):
    candidate_error = prompt_run(comparison.candidate, outcome="error")
    candidate_failure = prompt_run(comparison.candidate, outcome="failure")

    report, _ = comparison(prompt_run(comparison.baseline), candidate_error, candidate_failure)

    assert report["status"] == "incomplete"
    assert report["comparisons"][0]["candidate"] == "error"


@title("Test cases of other test modules are ignored")
def test_comparison_ignores_other_tests(comparison, tmp_path):
    path = write_junit(
            tmp_path / "other.xml",
            [prompt_run(comparison.baseline), prompt_run(comparison.candidate)], classname="tests.test_rag")

    report = compare_prompts(
            path, catalog_path=PROMPTS_FILE, dataset_path=GOLDEN_DATASET_FILE, policy_file=POLICY_FILE,
            case_ids=["paid_leave"], models=["model"], candidate="grounded_v2")

    assert len(report["missing_runs"]) == 2


@pytest.mark.parametrize(("selection", "message"), [
        pytest.param({"selected": "baseline"}, "candidate different from baseline", id="candidate-is-baseline"),
        pytest.param({"selected": "unknown"}, "candidate different from baseline", id="unknown-candidate"),
        pytest.param({"cases": ()}, "nonempty cases/models and positive repetitions", id="no-cases"),
        pytest.param({"models": ()}, "nonempty cases/models", id="no-models"),
        pytest.param({"repeat": 0}, "positive repetitions", id="zero-repetitions"),
        pytest.param({"repeat": True}, "positive repetitions", id="boolean-repetitions"),
        pytest.param({"cases": ("paid_leave", "paid_leave")}, "Duplicate case/model selection", id="duplicate-case"),
        pytest.param({"models": ("model", "model")}, "Duplicate case/model selection", id="duplicate-model"),
        pytest.param({"cases": ("unknown", )}, "Unknown golden case", id="unknown-case")])
@title("The comparison scope must be a nonempty, unique, known matrix with a non-baseline candidate [{param_id}]")
def test_comparison_rejects_invalid_scope(comparison, selection, message):
    with pytest.raises(ValueError, match=message):
        comparison(**selection)


@title("Every golden case may be declared at once")
def test_comparison_accepts_all_cases(comparison):
    all_cases = tuple(case.id for case in GOLDEN_DATASET.cases)

    report, _ = comparison(cases=all_cases)

    assert report["scope"]["expected_runs"] == 2 * len(all_cases)


@pytest.fixture
def cli(monkeypatch):
    compare = Mock(return_value={"status": "passed", "scope": {"expected_runs": 2}})
    monkeypatch.setattr(reporting, "compare_prompts", compare)

    def invoke(*arguments: str) -> int:
        monkeypatch.setattr("sys.argv", ["prompts", "runs.xml", "--case", "paid_leave", "--model", "model", *arguments])
        return main()

    return compare, invoke


@title("The CLI compares the declared scope with the reviewed inputs, saves the result and exits with 0 on a pass")
def test_cli_saves_comparison(cli, tmp_path, capsys):
    compare, invoke = cli
    output = tmp_path / "prompts.json"

    assert invoke("--output", str(output)) == 0

    assert json.loads(output.read_text()) == compare.return_value
    assert "Prompt comparison: passed; expected runs: 2" in capsys.readouterr().out
    compare.assert_called_once_with(
            Path("runs.xml"), catalog_path=Path("test_data/prompt-variants.json"),
            dataset_path=Path("test_data/golden-policy.json"), policy_file=Path("test_data/company-policy.txt"),
            case_ids=["paid_leave"], models=["model"], candidate="grounded_v2", repeat=1)


@title("The CLI passes repeated cases and models, the candidate, repetitions and custom inputs")
def test_cli_passes_options(cli, tmp_path):
    compare, invoke = cli

    invoke(
            "--output", str(tmp_path / "o.json"), "--case", "carryover_limit", "--model", "other", "--candidate",
            "other_prompt", "--repeat", "2", "--catalog", "c.json", "--dataset", "d.json", "--policy", "p.txt")

    assert compare.call_args.kwargs == {
            "catalog_path": Path("c.json"),
            "dataset_path": Path("d.json"),
            "policy_file": Path("p.txt"),
            "case_ids": ["paid_leave", "carryover_limit"],
            "models": ["model", "other"],
            "candidate": "other_prompt",
            "repeat": 2}


@title("A regression, failing baseline or incomplete comparison exits with 1")
def test_cli_reports_failure(cli, tmp_path):
    compare, invoke = cli
    compare.return_value = {"status": "regression", "scope": {"expected_runs": 2}}

    assert invoke("--output", str(tmp_path / "o.json")) == 1


@title("The CLI refuses to overwrite an earlier comparison")
def test_cli_refuses_existing_output(cli, tmp_path, capsys):
    compare, invoke = cli
    output = tmp_path / "o.json"
    output.write_text("earlier")

    with pytest.raises(SystemExit) as stopped:
        invoke("--output", str(output))

    assert stopped.value.code == 2
    assert "Output already exists" in capsys.readouterr().err
    compare.assert_not_called()


@pytest.mark.parametrize(
        "arguments", [
        pytest.param(["runs.xml", "--model", "model", "--output", "o.json"], id="without-case"),
        pytest.param(["runs.xml", "--case", "paid_leave", "--output", "o.json"], id="without-model"),
        pytest.param(["runs.xml", "--case", "paid_leave", "--model", "model"], id="without-output")])
@title("The CLI requires a case, a model and an output file [{param_id}]")
def test_cli_requires_scope_and_output(cli, monkeypatch, tmp_path, arguments):
    compare, _ = cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["prompts", *arguments])

    with pytest.raises(SystemExit) as stopped:
        main()

    assert stopped.value.code == 2
    compare.assert_not_called()


@title("Prompt regression collects an opt-in case/model/prompt matrix")
def test_selection_is_opt_in(framework_pytester):
    shutil.copytree(TEST_DATA, framework_pytester.path / "test_data")
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    framework_pytester.makepyfile(
            """
        import pytest
        @pytest.mark.rag
        @pytest.mark.prompt_regression
        def test_prompt(prompt_variant, golden_case, generation_model, rag_iteration):
            assert golden_case.id == 'carryover_limit'
            assert prompt_variant.id in ('baseline', 'grounded_v2')
    """)
    arguments = ("-q", "-k", "carryover_limit", "--rag-model", "test")

    skipped = framework_pytester.runpytest_subprocess(*arguments)
    enabled = framework_pytester.runpytest_subprocess(*arguments, "--run-prompt-regression")

    skipped.assert_outcomes(skipped=2, deselected=30)
    enabled.assert_outcomes(passed=2, deselected=30)
