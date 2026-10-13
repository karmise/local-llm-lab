"""Stability summary: repeated RAG runs grouped by scenario and model, never treated as retries."""

import json
import sys

import pytest

from llm_testkit.reporting import stability
from llm_testkit.reporting.stability import summarize_report
from llm_testkit.reporting.steps import title
from test_support.builders.junit_xml import write_junit

pytestmark = pytest.mark.unit

RAG = "tests.test_rag"
CONFIGURATION = json.dumps({"chatModel": "qwen", "openAiPrompt": "Policy"})


def run(iteration: int = 1, outcome: str | None = None, *, name: str = "test_example", **changes) -> dict:
    """One repetition of a RAG test with complete metadata; a property set to None is dropped."""
    properties = {
            "generation_model": "qwen",
            "model_digest": "digest",
            "policy_sha256": "policy",
            "workspace_configuration": CONFIGURATION,
            "thinking_mode": "default",
            **changes}
    return {
            "name": f"{name}[qwen-run-{iteration}]",
            "outcome": outcome,
            "properties": {
            k: v
            for k, v in properties.items() if v is not None}}


@pytest.fixture
def summarize(tmp_path):
    """Summarize a JUnit report written from the given runs."""
    def summarize(*runs):
        return summarize_report(write_junit(tmp_path / "report.xml", list(runs), classname=RAG))

    return summarize


@title("Repetitions of one scenario on one model form one group with complete, consistent metadata")
def test_summary_groups_repetitions(summarize):
    rows = summarize(run(1), run(2), run(3))

    assert rows == [{
            "classname": RAG,
            "scenario": "test_example",
            "model": "qwen",
            "runs": 3,
            "passed": 3,
            "failed": 0,
            "errored": 0,
            "skipped": 0,
            "metadata_complete": True,
            "configuration_consistent": True,
            "mixed_pass_fail_observed": False}]


@title("Mixed passed and failed repetitions are reported, not retried away")
def test_summary_preserves_mixed_pass_fail_results(summarize):
    row, = summarize(run(1), run(2, "failure"), run(3))

    assert (row["runs"], row["passed"], row["failed"], row["mixed_pass_fail_observed"]) == (3, 2, 1, True)


@pytest.mark.parametrize(("outcome", "counted"), [
        pytest.param("failure", "failed", id="failure"),
        pytest.param("error", "errored", id="error"),
        pytest.param("skipped", "skipped", id="skipped")])
@title("Each outcome is counted in its own column and is not a pass [{param_id}]")
def test_summary_counts_outcomes(summarize, outcome, counted):
    row, = summarize(run(1, outcome), run(2, outcome), run(3))

    assert {
            k: row[k]
            for k in ("passed", "failed", "errored", "skipped")} == {
            "passed": 1,
            "failed": 0,
            "errored": 0,
            "skipped": 0} | {
            counted: 2}


@title("One pass next to one failure is already mixed")
def test_single_pass_and_failure_are_mixed(summarize):
    row, = summarize(run(1), run(2, "failure"))

    assert row["mixed_pass_fail_observed"] is True


@pytest.mark.parametrize("outcome", ["error", "skipped"])
@title("Only a pass next to a failure is mixed; errors and skips are not model failures [{outcome}]")
def test_summary_mixed_requires_a_failure(summarize, outcome):
    row, = summarize(run(1), run(2, outcome))

    assert row["mixed_pass_fail_observed"] is False


@title("A setup error without metadata is not a model failure and leaves metadata incomplete")
def test_setup_error_without_metadata(summarize):
    unrecorded = run(
            2, "error", **dict.fromkeys(("model_digest", "policy_sha256", "workspace_configuration", "thinking_mode")))

    row, = summarize(run(1), unrecorded)

    assert (row["runs"], row["errored"], row["failed"]) == (2, 1, 0)
    assert (row["metadata_complete"], row["configuration_consistent"]) == (False, False)


@title("Call and teardown entries of one test count as one run with both outcomes")
def test_call_and_teardown_entries_count_as_one_run(summarize):
    row, = summarize(run(1, "failure"), {"name": "test_example[qwen-run-1]", "properties": {}, "outcome": "error"})

    assert (row["runs"], row["failed"], row["errored"], row["passed"]) == (1, 1, 1, 0)


@title("Without a generation_model property the model is read from a RAG test's parameter id")
def test_model_falls_back_to_rag_parameter_id(summarize):
    rows = summarize(
            run(1, generation_model=None), run(2, generation_model=None), {
            "name": "test_single[llama]",
            "properties": {},
            "outcome": None})

    assert [(row["scenario"], row["model"], row["runs"]) for row in rows] == [("test_example", "qwen", 2),
            ("test_single", "llama", 1)]


@title("Tests outside RAG modules need an explicit generation model to be summarized")
def test_non_rag_test_without_model_is_ignored(summarize):
    other = run(1, generation_model=None) | {"classname": "tests.test_ui"}
    labelled = run(2) | {"classname": "tests.test_ui"}

    rows = summarize(other, labelled, {"name": "test_plain", "properties": {}, "outcome": None})

    assert [(row["classname"], row["runs"]) for row in rows] == [("tests.test_ui", 1)]


@title("Equal function names in different modules are distinct scenarios")
def test_equal_names_in_different_modules_are_distinct(summarize):
    rows = summarize(run(1), run(2, "failure") | {"classname": "tests.test_other_rag"})

    assert [(row["classname"], row["runs"], row["mixed_pass_fail_observed"]) for row in rows] == [
            ("tests.test_other_rag", 1, False), (RAG, 1, False)]


@title("Different generation models are separate groups")
def test_models_are_separate_groups(summarize):
    rows = summarize(run(1), run(2, "failure", generation_model="llama"))

    assert [(row["model"], row["runs"]) for row in rows] == [("llama", 1), ("qwen", 1)]


@pytest.mark.parametrize(("properties", "scenario"), [
        pytest.param({
        "golden_case_id": "paid_leave",
        "golden_dataset_sha256": "d"}, "test_example[paid_leave]", id="golden"),
        pytest.param({
        "bias_pair_id": "gender",
        "bias_variant_id": "2",
        "bias_catalog_sha256": "b"}, "test_example[bias=gender:2]", id="bias"),
        pytest.param({
        "adversarial_case_id": "override",
        "adversarial_catalog_sha256": "a"}, "test_example[attack=override]", id="adversarial"),
        pytest.param({
        "conversation_case_id": "greeting",
        "conversation_catalog_sha256": "c"}, "test_example[conversation=greeting]", id="conversation"),
        pytest.param({
        "prompt_id": "grounded_v2",
        "prompt_sha256": "p"}, "test_example[prompt=grounded_v2]", id="prompt"),
        pytest.param({
        "golden_case_id": "paid_leave",
        "golden_dataset_sha256": "d",
        "prompt_id": "baseline",
        "prompt_sha256": "p"}, "test_example[paid_leave][prompt=baseline]", id="golden-and-prompt")])
@title("Scenario identity includes the catalog case it ran [{param_id}]")
def test_scenario_identity(summarize, properties, scenario):
    row, = summarize(run(1, **properties))

    assert (row["scenario"], row["metadata_complete"], row["configuration_consistent"]) == (scenario, True, True)


@title("A bias run without a variant is labelled as missing and is incomplete")
def test_bias_run_without_variant(summarize):
    row, = summarize(run(1, bias_pair_id="gender", bias_catalog_sha256="b"))

    assert (row["scenario"], row["metadata_complete"]) == ("test_example[bias=gender:missing]", False)


@pytest.mark.parametrize(("first", "second"), [
        pytest.param({"golden_case_id": "first"}, {"golden_case_id": "second"}, id="golden-cases"),
        pytest.param({"prompt_id": "baseline"}, {"prompt_id": "grounded_v2"}, id="prompt-variants"),
        pytest.param({
        "bias_pair_id": "gender",
        "bias_variant_id": "1"}, {
        "bias_pair_id": "gender",
        "bias_variant_id": "2"}, id="bias-variants"),
        pytest.param({"adversarial_case_id": "a"}, {"adversarial_case_id": "b"}, id="attacks"),
        pytest.param({"conversation_case_id": "greeting"}, {"conversation_case_id": "mixed_request"},
        id="conversations")])
@title("Different catalog cases are separate scenarios, not flaky repetitions [{param_id}]")
def test_catalog_cases_are_separate_scenarios(summarize, first, second):
    rows = summarize(run(1, **first), run(2, "failure", **second))

    assert [(row["runs"], row["mixed_pass_fail_observed"]) for row in rows] == [(1, False), (1, False)]


METADATA_FIELDS = ["model_digest", "policy_sha256", "workspace_configuration", "thinking_mode"]


@pytest.mark.parametrize(("properties", "missing"), [
        *(pytest.param({}, field, id=field) for field in METADATA_FIELDS),
        pytest.param({"golden_case_id": "paid_leave"}, "golden_dataset_sha256", id="golden-dataset"),
        pytest.param({
        "bias_pair_id": "gender",
        "bias_variant_id": "1"}, "bias_catalog_sha256", id="bias-catalog"),
        pytest.param({"adversarial_case_id": "override"}, "adversarial_catalog_sha256", id="adversarial-catalog"),
        pytest.param({"conversation_case_id": "greeting"}, "conversation_catalog_sha256", id="conversation-catalog"),
        pytest.param({"prompt_id": "baseline"}, "prompt_sha256", id="prompt-checksum")])
@title("A run without its required provenance makes the group's metadata incomplete [{param_id}]")
def test_missing_provenance_is_incomplete(summarize, properties, missing):
    row, = summarize(run(1, **properties, **{missing: ""}))

    assert (row["metadata_complete"], row["configuration_consistent"]) == (False, False)


@pytest.mark.parametrize(
        "configuration", [
        pytest.param("not-json", id="invalid-json"),
        pytest.param("[]", id="not-object"),
        pytest.param(json.dumps({"openAiPrompt": 1}), id="non-string-prompt")])
@title("An unreadable workspace configuration makes metadata incomplete [{param_id}]")
def test_invalid_configuration_is_incomplete(summarize, configuration):
    row, = summarize(run(1), run(2, workspace_configuration=configuration))

    assert (row["metadata_complete"], row["configuration_consistent"]) == (False, False)


@title("Conflicting values for one property within a run make metadata incomplete")
def test_conflicting_properties_are_incomplete(summarize):
    teardown = {"name": "test_example[qwen-run-1]", "properties": {"model_digest": "stale"}, "outcome": None}

    row, = summarize(run(1), teardown)

    assert (row["runs"], row["metadata_complete"]) == (1, False)


@pytest.mark.parametrize(
        "changes", [
        pytest.param({"model_digest": "other"}, id="model-digest"),
        pytest.param({"policy_sha256": "other"}, id="policy"),
        pytest.param({"workspace_configuration": json.dumps({
        "chatModel": "qwen",
        "topN": 2})}, id="configuration"),
        pytest.param({"thinking_mode": "think"}, id="thinking"),
        pytest.param({"golden_dataset_sha256": "other"}, id="golden-dataset"),
        pytest.param({"prompt_sha256": "other"}, id="prompt"),
        pytest.param({"adversarial_catalog_sha256": "other"}, id="adversarial-catalog"),
        pytest.param({"bias_catalog_sha256": "other"}, id="bias-catalog"),
        pytest.param({"conversation_catalog_sha256": "other"}, id="conversation-catalog")])
@title("Repetitions run under different conditions are complete but not consistent [{param_id}]")
def test_changed_conditions_are_inconsistent(summarize, changes):
    row, = summarize(run(1), run(2, **changes))

    assert (row["metadata_complete"], row["configuration_consistent"]) == (True, False)


@title("Per-run capture markers and key order do not change the configuration fingerprint")
def test_configuration_is_normalized(summarize):
    first = json.dumps({"openAiPrompt": f"Policy\n[LLM_TESTKIT_CAPTURE:{'a' * 32}]", "chatModel": "qwen"})
    second = json.dumps({"chatModel": "qwen", "openAiPrompt": f"Policy\n[LLM_TESTKIT_CAPTURE:{'b' * 32}]"})

    row, = summarize(run(1, workspace_configuration=first), run(2, workspace_configuration=second))

    assert (row["metadata_complete"], row["configuration_consistent"]) == (True, True)


@title("Groups are sorted by module, scenario and model")
def test_summary_is_sorted(summarize):
    rows = summarize(
            run(1, name="test_b"), run(1, name="test_a", generation_model="z"), run(2, name="test_a"),
            run(1) | {"classname": "tests.test_a_rag"})

    assert [(row["classname"], row["scenario"], row["model"]) for row in rows] == [
            ("tests.test_a_rag", "test_example", "qwen"), (RAG, "test_a", "qwen"), (RAG, "test_a", "z"),
            (RAG, "test_b", "qwen")]


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["stability", *map(str, args)])
    return stability.main()


@title("The command prints the summary and can also write it, creating the output directory")
def test_cli_writes_summary(tmp_path, monkeypatch, capsys):
    report = write_junit(tmp_path / "report.xml", [run(1)], classname=RAG)
    output = tmp_path / "reports/stability/summary.json"

    run_cli(monkeypatch, report, "--output", output)

    printed = capsys.readouterr().out
    assert printed == output.read_text(encoding="utf-8") == json.dumps(summarize_report(report), indent=2) + "\n"


@title("The command prints the summary without writing a file by default")
def test_cli_prints_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    report = write_junit(tmp_path / "report.xml", [run(1)], classname=RAG)

    run_cli(monkeypatch, report)

    assert json.loads(capsys.readouterr().out)[0]["runs"] == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.xml"]


@title("A report without identifiable RAG cases is a usage error")
def test_cli_rejects_report_without_rag_cases(tmp_path, monkeypatch, capsys):
    report = write_junit(
            tmp_path / "report.xml", [{
            "name": "test_plain",
            "properties": {},
            "outcome": None}], classname="tests.test_ui")

    with pytest.raises(SystemExit) as exit:
        run_cli(monkeypatch, report)

    assert exit.value.code == 2
    assert "The report contains no identifiable RAG cases" in capsys.readouterr().err


@title("The command writes into an output directory that already exists")
def test_cli_writes_into_existing_directory(tmp_path, monkeypatch, capsys):
    report = write_junit(tmp_path / "report.xml", [run(1)], classname=RAG)

    run_cli(monkeypatch, report, "--output", tmp_path / "summary.json")

    assert json.loads((tmp_path / "summary.json").read_text())[0]["runs"] == 1
