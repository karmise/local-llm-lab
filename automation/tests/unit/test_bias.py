"""Counterfactual bias pairs: a reviewed catalog, answer criteria, paired JUnit analysis, the CLI and selection."""

import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pytest

from llm_testkit import assertions
from llm_testkit.datasets.bias import load_bias_cases
from llm_testkit.reporting import bias as reporting
from llm_testkit.reporting.bias import compare_pairs, main
from llm_testkit.reporting.steps import title
from test_support.builders.anythingllm import chat_reply
from test_support.builders.bias import BIAS_FILE, SETTINGS, bias_cases, bias_run, catalog_json, write_junit
from test_support.builders.golden import GOLDEN_DATASET, GOLDEN_DATASET_FILE, POLICY_FILE, TEST_DATA
from test_support.data.common import POLICY_DOCUMENT_TITLE

pytestmark = pytest.mark.unit

PAIRS = ["gender_carryover", "age_carryover", "nationality_carryover"]
GENDER = "gender_carryover"


def load(data: dict, tmp_path):
    path = tmp_path / "bias.json"
    path.write_text(json.dumps(data))
    return load_bias_cases(path, GOLDEN_DATASET)


def check_answer(case, answer: str) -> None:
    assertions.assert_bias_answer(chat_reply(answer), case=case, document_title=POLICY_DOCUMENT_TITLE)


@title("The reviewed catalog loads two variants per pair that differ only in the employee descriptor")
def test_catalog_loads_pairs():
    raw = BIAS_FILE.read_bytes()

    cases = load_bias_cases(BIAS_FILE, GOLDEN_DATASET)

    assert [(case.pair_id, case.variant_id) for case in cases] == [(p, v) for p in PAIRS for v in ("1", "2")]
    first, second = cases[:2]
    assert (first.attribute, first.descriptor, second.descriptor) == ("gender", "a woman", "a man")
    assert first.question == "I am an employee and I am a woman. " + first.golden_case.question
    assert first.golden_case.id == "carryover_limit"
    assert dict(first.forbidden_patterns).keys() == {"unsupported eligibility restriction", "stereotyped capability"}
    assert (first.catalog_sha256, first.catalog_version) == (hashlib.sha256(raw).hexdigest(), catalog_json()["version"])


def gender(**fields):
    return lambda data: data["pairs"][0].update(fields)


def variant(index: int, **fields):
    return lambda data: data["pairs"][0]["variants"][index].update(fields)


@pytest.mark.parametrize(("corrupt", "message"), [
        pytest.param(lambda d: d.update(schema_version=2), "Unsupported bias catalog schema", id="schema"),
        pytest.param(
        lambda d: d.update(golden_dataset_sha256="changed"), "bind to the current golden dataset",
        id="stale-golden-dataset"),
        pytest.param(lambda d: d.pop("version"), "bias catalog version must be", id="no-version"),
        pytest.param(lambda d: d.update(pairs=[]), "requires pairs", id="no-pairs"),
        pytest.param(lambda d: d.update(pairs={}), "requires pairs", id="pairs-not-a-list"),
        pytest.param(lambda d: d["pairs"].__setitem__(0, "pair"), "row must be an object", id="pair-not-object"),
        pytest.param(gender(id="Gender"), "Invalid or duplicate bias pair", id="invalid-id"),
        pytest.param(gender(id=None), "Invalid or duplicate bias pair", id="no-id"),
        pytest.param(lambda d: d["pairs"][1].update(id=GENDER), "Invalid or duplicate bias pair", id="duplicate-id"),
        pytest.param(gender(attribute="religion"), "Unknown bias attribute", id="unknown-attribute"),
        pytest.param(
        gender(golden_case_id="unknown"), "Unknown bias attribute or golden expectation", id="unknown-golden-case"),
        pytest.param(gender(golden_case_id=None), "Unknown bias attribute or golden expectation", id="no-golden-case"),
        pytest.param(gender(rationale=" "), "requires reviewed rationale", id="blank-rationale"),
        pytest.param(gender(rationale=None), "requires reviewed rationale", id="no-rationale"),
        pytest.param(lambda d: d["pairs"][0]["variants"].pop(), "exactly two distinct variants", id="one-variant"),
        pytest.param(gender(variants={}), "exactly two distinct variants", id="variants-not-a-list"),
        pytest.param(
        lambda d: d["pairs"][0]["variants"].__setitem__(1, "a man"), "exactly two distinct variants",
        id="variant-not-object"),
        pytest.param(variant(1, id="1"), "exactly two distinct variants", id="same-variant-ids"),
        pytest.param(variant(1, id=2), "exactly two distinct variants", id="numeric-variant-id"),
        pytest.param(variant(1, descriptor="a woman"), "descriptors must differ", id="same-descriptor"),
        pytest.param(
        gender(forbidden_patterns={}), "forbidden_patterns must be a nonempty object", id="no-forbidden-rules"),
        pytest.param(
        gender(forbidden_patterns={"empty": ".*"}), "gender_carryover: forbidden_patterns.empty", id="vacuous-rule"),
        pytest.param(variant(0, descriptor=" "), "Only the declared descriptor may change", id="blank-descriptor"),
        pytest.param(variant(0, descriptor=None), "Only the declared descriptor may change", id="no-descriptor"),
        pytest.param(
        lambda d: d["pairs"][0]["variants"][0].update(
        question=d["pairs"][0]["variants"][0]["question"] + " Another demand."),
        "Only the declared descriptor may change", id="question-changed")])
@title("The catalog lets only the reviewed descriptor change between the two variants [{param_id}]")
def test_catalog_rejects_invalid_pair(tmp_path, corrupt, message):
    data = catalog_json()
    corrupt(data)

    with pytest.raises(ValueError, match=message):
        load(data, tmp_path)


@pytest.mark.parametrize("pair", PAIRS)
@title("A grounded answer passes; an unsupported eligibility restriction fails for every pair [{param_id}]")
def test_answer_rejects_eligibility_restriction(pair):
    case = bias_cases()[(pair, "1")]

    check_answer(case, case.golden_case.reference)
    with pytest.raises(AssertionError, match="Bias criterion failed: unsupported eligibility restriction"):
        check_answer(case, case.golden_case.reference + " You are NOT ELIGIBLE.")


@title("An answer that ascribes a stereotyped capability fails")
def test_answer_rejects_stereotype():
    case = bias_cases()[(GENDER, "1")]

    with pytest.raises(AssertionError, match="Bias criterion failed: stereotyped capability"):
        check_answer(case, case.golden_case.reference + " Women are less capable of planning leave.")


class PairedRuns:
    """The reviewed cases and a compare function, with shortcuts for the two gender variants."""
    def __init__(self, cases, compare):
        self.cases, self.compare = cases, compare

    def run(self, variant_id: str, **changes):
        return bias_run(self.cases[(GENDER, variant_id)], **changes)


@pytest.fixture
def paired_report(tmp_path):
    """Write JUnit evidence for the gender pair and compare it for one model and one repetition."""
    cases = bias_cases()

    def compare(*runs, pairs=(GENDER, ), models=("model", ), repeat=1):
        if not runs:
            runs = (bias_run(cases[(GENDER, "1")]), bias_run(cases[(GENDER, "2")]))
        path = write_junit(tmp_path / "results.xml", list(runs))
        report = compare_pairs(
                path, catalog=BIAS_FILE, dataset_path=GOLDEN_DATASET_FILE, policy=POLICY_FILE, pair_ids=list(pairs),
                models=list(models), repeat=repeat)
        return report, path

    return PairedRuns(cases, compare)


@title("Two passing variants with identical settings are a passing comparison within the declared scope")
def test_comparison_passes(paired_report):
    report, path = paired_report.compare()

    assert (report["schema_version"], report["status"]) == (1, "passed")
    assert report["report_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["scope"] == {"pairs": [GENDER], "models": ["model"], "repeat": 1, "expected_runs": 2}
    assert (report["missing_runs"], report["errors"]) == ([], [])
    assert report["comparisons"] == [{"pair": GENDER, "model": "model", "iteration": 1, "outcome": "passed"}]
    assert "not a demographic fairness estimate" in report["interpretation"]


@pytest.mark.parametrize(("outcomes", "status", "outcome"), [
        pytest.param((None, "failure"), "failed", "asymmetry", id="one-variant-failed"),
        pytest.param(("failure", "failure"), "failed", "shared_failure", id="both-failed"),
        pytest.param((None, "error"), "incomplete", "incomplete", id="one-variant-error"),
        pytest.param(("skipped", None), "incomplete", "incomplete", id="one-variant-skipped")])
@title("Pair outcomes distinguish asymmetry, shared failures and incomplete evidence [{param_id}]")
def test_comparison_outcomes(paired_report, outcomes, status, outcome):
    report, _ = paired_report.compare(
            paired_report.run("1", outcome=outcomes[0]), paired_report.run("2", outcome=outcomes[1]))

    assert (report["status"], report["comparisons"][0]["outcome"]) == (status, outcome)


@title("A missing variant leaves the comparison incomplete and is listed as missing")
def test_comparison_reports_missing_run(paired_report):
    report, _ = paired_report.compare(paired_report.run("1"))

    assert report["status"] == "incomplete"
    assert report["missing_runs"] == [(GENDER, "model", 1, "2")]


@pytest.mark.parametrize(("second", "error"), [
        pytest.param({"generation_model": "other"}, "Unexpected paired run", id="unplanned-model"),
        pytest.param({"rag_iteration": "2"}, "Unexpected paired run", id="unplanned-repetition"),
        pytest.param({"rag_iteration": "first"}, "invalid literal", id="non-numeric-repetition"),
        pytest.param({"bias_catalog_sha256": "stale"}, "Stale paired expectations", id="stale-catalog"),
        pytest.param({"golden_dataset_sha256": "stale"}, "Stale paired expectations", id="stale-dataset"),
        pytest.param({"policy_sha256": "stale"}, "Stale paired expectations", id="stale-policy"),
        pytest.param({"golden_case_id": "paid_leave"}, "Stale paired expectations", id="other-golden-case"),
        pytest.param({"workspace_configuration": json.dumps({"chatModel": "model"})}, "Missing prompt configuration",
        id="no-prompt"),
        pytest.param({"workspace_configuration": json.dumps({
        **SETTINGS, "chatModel": "other"})}, "inconsistent model metadata", id="other-chat-model"),
        pytest.param({"model_digest": ""}, "inconsistent model metadata", id="no-digest"),
        pytest.param({"thinking_mode": ""}, "inconsistent model metadata", id="no-thinking-mode"),
        pytest.param({"bias_pair_id": None}, "'bias_pair_id'", id="no-pair-id"),
        pytest.param({"workspace_configuration": "[]"}, "must be an object", id="settings-not-object")])
@title("Runs that are duplicated, unplanned, stale or lack model metadata make the comparison incomplete [{param_id}]")
def test_comparison_rejects_inconsistent_runs(paired_report, second, error):
    report, _ = paired_report.compare(paired_report.run("1"), paired_report.run("2", **second))

    assert report["status"] == "incomplete"
    assert any(error in message for message in report["errors"])


@title("A duplicated run removes both copies, so neither is used for the comparison")
def test_comparison_discards_duplicated_run(paired_report):
    report, _ = paired_report.compare(
            paired_report.run("1"), paired_report.run("1", name="again"), paired_report.run("1", name="third"),
            paired_report.run("2"))

    assert report["missing_runs"] == [(GENDER, "model", 1, "1")]
    assert report["errors"] == ["Duplicate paired run", "Duplicate paired run"]


@title("Conflicting metadata within one run is reported and the run is not used")
def test_comparison_rejects_conflicting_metadata(paired_report, tmp_path):
    path = write_junit(tmp_path / "results.xml", [paired_report.run("1"), paired_report.run("2")])
    conflicting = '<property name="model_digest" value="x"/><property name="model_digest"'
    path.write_text(path.read_text().replace('<property name="model_digest"', conflicting, 1))

    report = compare_pairs(
            path, catalog=BIAS_FILE, dataset_path=GOLDEN_DATASET_FILE, policy=POLICY_FILE, pair_ids=[GENDER],
            models=["model"])

    assert report["status"] == "incomplete"
    assert report["errors"] == ["Conflicting metadata: ['model_digest']"]


@title("Variants generated with different weights or settings cannot be compared")
def test_comparison_rejects_changed_configuration(paired_report):
    report, _ = paired_report.compare(paired_report.run("1"), paired_report.run("2", model_digest="other"))

    assert report["status"] == "incomplete"
    assert report["errors"] == ["Paired model/prompt/retrieval configuration changed"]
    assert report["comparisons"][0]["outcome"] == "incomplete"


@title("The per-request capture marker in the prompt does not count as a configuration change")
def test_comparison_ignores_capture_marker(paired_report):
    marked = json.dumps({**SETTINGS, "openAiPrompt": "Policy\n[LLM_TESTKIT_CAPTURE:" + "a" * 32 + "]"})

    report, _ = paired_report.compare(paired_report.run("1"), paired_report.run("2", workspace_configuration=marked))

    assert report["status"] == "passed"


@title("Test cases from other test modules are ignored")
def test_comparison_ignores_other_tests(paired_report, tmp_path):
    path = write_junit(tmp_path / "other.xml", [paired_report.run("1"), paired_report.run("2")], classname="tests.x")

    report = compare_pairs(
            path, catalog=BIAS_FILE, dataset_path=GOLDEN_DATASET_FILE, policy=POLICY_FILE, pair_ids=[GENDER],
            models=["model"])

    assert report["missing_runs"] == [(GENDER, "model", 1, "1"), (GENDER, "model", 1, "2")]


@title("Repetitions multiply the expected runs; each repetition is compared separately")
def test_comparison_with_repetitions(paired_report):
    runs = [paired_report.run(v, rag_iteration=str(i), name=f"test_bias[{v}-{i}]") for i in (1, 2) for v in ("1", "2")]
    runs[-1]["outcome"] = "failure"

    report, _ = paired_report.compare(*runs, repeat=2)

    assert report["scope"]["expected_runs"] == 4
    assert [c["outcome"] for c in report["comparisons"]] == ["passed", "asymmetry"]


@pytest.mark.parametrize(("selection", "message"), [
        pytest.param({"pairs": ()}, "Declare known pairs, models and positive repetitions", id="no-pairs"),
        pytest.param({"models": ()}, "Declare known pairs", id="no-models"),
        pytest.param({"pairs": ("unknown", )}, "Declare known pairs", id="unknown-pair"),
        pytest.param({"repeat": 0}, "Declare known pairs", id="zero-repetitions"),
        pytest.param({"repeat": True}, "Declare known pairs", id="boolean-repetitions"),
        pytest.param({"pairs": (GENDER, GENDER)}, "Duplicate selection", id="duplicate-pair"),
        pytest.param({"models": ("model", "model")}, "Duplicate selection", id="duplicate-model")])
@title("The comparison scope must name known pairs and models once, with positive repetitions [{param_id}]")
def test_comparison_rejects_invalid_scope(paired_report, selection, message):
    with pytest.raises(ValueError, match=message):
        paired_report.compare(**selection)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    comparison = Mock(return_value={"status": "passed"})
    monkeypatch.setattr(reporting, "compare_pairs", comparison)

    def invoke(*arguments: str) -> int:
        monkeypatch.setattr("sys.argv", ["bias", "results.xml", "--pair", GENDER, "--model", "model", *arguments])
        return main()

    return comparison, invoke


@title("The CLI compares the declared scope with the reviewed inputs, saves the result and exits with 0 on a pass")
def test_cli_saves_comparison(cli, tmp_path, capsys):
    comparison, invoke = cli
    output = tmp_path / "bias.json"

    assert invoke("--output", str(output)) == 0

    assert json.loads(output.read_text()) == {"status": "passed"}
    assert "Bias acceptance comparison: passed" in capsys.readouterr().out
    comparison.assert_called_once_with(
            Path("results.xml"), catalog=Path("test_data/bias-policy.json"),
            dataset_path=Path("test_data/golden-policy.json"), policy=Path("test_data/company-policy.txt"),
            pair_ids=[GENDER], models=["model"], repeat=1)


@title("The CLI passes repeated pairs, models, repetitions and custom inputs")
def test_cli_passes_options(cli, tmp_path):
    comparison, invoke = cli

    invoke(
            "--output", str(tmp_path / "o.json"), "--pair", "age_carryover", "--model", "other", "--repeat", "3",
            "--catalog", "c.json", "--dataset", "d.json", "--policy", "p.txt")

    assert comparison.call_args.kwargs == {
            "catalog": Path("c.json"),
            "dataset_path": Path("d.json"),
            "policy": Path("p.txt"),
            "pair_ids": [GENDER, "age_carryover"],
            "models": ["model", "other"],
            "repeat": 3}


@title("A failed or incomplete comparison exits with 1")
def test_cli_reports_failure(cli, tmp_path):
    comparison, invoke = cli
    comparison.return_value = {"status": "incomplete"}

    assert invoke("--output", str(tmp_path / "o.json")) == 1


@title("The CLI refuses to overwrite an earlier comparison")
def test_cli_refuses_existing_output(cli, tmp_path, capsys):
    comparison, invoke = cli
    output = tmp_path / "o.json"
    output.write_text("earlier")

    with pytest.raises(SystemExit) as stopped:
        invoke("--output", str(output))

    assert stopped.value.code == 2
    assert "Output already exists" in capsys.readouterr().err
    comparison.assert_not_called()


@pytest.mark.parametrize(
        "arguments", [
        pytest.param(["results.xml", "--model", "model", "--output", "o.json"], id="without-pair"),
        pytest.param(["results.xml", "--pair", GENDER, "--output", "o.json"], id="without-model"),
        pytest.param(["results.xml", "--pair", GENDER, "--model", "model"], id="without-output")])
@title("The CLI requires a pair, a model and an output file [{param_id}]")
def test_cli_requires_scope_and_output(cli, monkeypatch, tmp_path, arguments):
    comparison, _ = cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["bias", *arguments])

    with pytest.raises(SystemExit) as stopped:
        main()

    assert stopped.value.code == 2
    comparison.assert_not_called()


@title("Bias variants collect as independent opt-in case/model runs")
def test_selection_is_opt_in(framework_pytester):
    shutil.copytree(TEST_DATA, framework_pytester.path / "test_data")
    framework_pytester.makeconftest('pytest_plugins=["llm_testkit.pytest_support.options"]')
    framework_pytester.makepyfile(
            """
        import pytest
        @pytest.mark.bias
        @pytest.mark.rag
        def test_pair(bias_case, generation_model, rag_iteration):
            assert bias_case.pair_id == 'gender_carryover'
    """)

    skipped = framework_pytester.runpytest_subprocess("-q", "--rag-model", "test", "-k", "gender")
    enabled = framework_pytester.runpytest_subprocess("-q", "--rag-model", "test", "-k", "gender", "--run-bias")

    skipped.assert_outcomes(skipped=2, deselected=4)
    enabled.assert_outcomes(passed=2, deselected=4)


@title("Every reviewed pair may be selected at once; pairs without runs are reported as missing")
def test_comparison_accepts_all_pairs(paired_report):
    report, _ = paired_report.compare(pairs=PAIRS)

    assert report["scope"]["expected_runs"] == 6
    assert [c["outcome"] for c in report["comparisons"]] == ["passed", "incomplete", "incomplete"]
