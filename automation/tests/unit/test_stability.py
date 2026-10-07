import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from llm_testkit.reporting.stability import summarize_report
from llm_testkit.reporting.steps import title
from test_support.builders.stability import (
    _report,
    check_conversation_summary_outcome,
    check_golden_summary_outcome,
    prepare_bias_groups_step_3,
    prepare_configuration_metadata_step_3,
    prepare_conversation_summary_step_3,
    prepare_golden_summary_step_3,
    prepare_prompt_variants_are_not_reported_as_flaky_repetitions_step_3,
)
from test_support.data.stability import (
    CONFIGURATION_METADATA_CHANGE_CASES,
    CONVERSATION_SUMMARY_CHANGE_CASES,
    GOLDEN_SUMMARY_SAME_CASE_CASES,
    GOLDEN_SUMMARY_SAME_CASE_IDS,
)

pytestmark = pytest.mark.unit


@title("Stability summary preserves mixed passed and failed runs")
def test_summary_preserves_mixed_pass_fail_results(tmp_path: Path) -> None:
    row = summarize_report(_report(tmp_path, ["passed", "failure", "passed"]))[0]
    assert row["runs"] == 3
    assert row["passed"] == 2
    assert row["failed"] == 1
    assert row["mixed_pass_fail_observed"] is True
    assert row["configuration_consistent"] is True


@title("Stability summary detects changes in model configuration")
def test_summary_detects_configuration_changes(tmp_path: Path) -> None:
    row = summarize_report(_report(tmp_path, ["passed", "failure"], ["first", "second"]))[0]
    assert row["configuration_consistent"] is False


@title("Stability summary distinguishes setup errors from model failures")
def test_setup_error_without_metadata_is_not_a_model_failure(tmp_path: Path) -> None:
    path = _report(tmp_path, ["passed", "error"])
    tree = ET.parse(path)
    failed_case = tree.getroot().findall(".//testcase")[1]
    failed_case.remove(failed_case.find("properties"))
    tree.write(path)
    row = summarize_report(path)[0]
    assert row["runs"] == 2
    assert row["errored"] == 1
    assert row["failed"] == 0
    assert row["mixed_pass_fail_observed"] is False
    assert row["metadata_complete"] is False


@title("Stability summary counts call and teardown entries as one run")
def test_duplicate_call_and_teardown_entries_count_as_one_run(tmp_path: Path) -> None:
    path = _report(tmp_path, ["failure"])
    tree = ET.parse(path)
    suite = tree.getroot().find("testsuite")
    duplicate = ET.SubElement(
        suite,
        "testcase",
        {
            "classname": "tests.test_rag",
            "name": "test_example[qwen-run-1]",
        },
    )
    ET.SubElement(duplicate, "error")
    tree.write(path)
    row = summarize_report(path)[0]
    assert row["runs"] == 1
    assert row["failed"] == 1
    assert row["errored"] == 1


@pytest.mark.parametrize(
    "same_case",
    GOLDEN_SUMMARY_SAME_CASE_CASES,
    ids=GOLDEN_SUMMARY_SAME_CASE_IDS,
)
@title("Golden stability preserves scenario identity and expectation fingerprints [{param_id}]")
def test_golden_summary_preserves_case_and_dataset_identity(
    tmp_path: Path, same_case: bool
) -> None:
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    prepare_golden_summary_step_3(same_case, tree)
    tree.write(path)
    rows = summarize_report(path)
    check_golden_summary_outcome(rows, same_case)


@title("Stability summaries keep different prompt variants in separate groups")
def test_prompt_variants_are_not_reported_as_flaky_repetitions(tmp_path: Path) -> None:
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    prepare_prompt_variants_are_not_reported_as_flaky_repetitions_step_3(tree)
    tree.write(path)
    rows = summarize_report(path)
    assert len(rows) == 2
    assert all(not r["mixed_pass_fail_observed"] for r in rows)


@title("Counterfactual variants remain separate stability scenarios")
def test_bias_groups(tmp_path):
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    prepare_bias_groups_step_3(tree)
    tree.write(path)
    assert len(summarize_report(path)) == 2


@pytest.mark.parametrize("change", CONFIGURATION_METADATA_CHANGE_CASES)
def test_configuration_metadata_is_normalized_and_validated(tmp_path, change):

    path = _report(tmp_path, ["passed", "passed"])
    tree = ET.parse(path)
    prepare_configuration_metadata_step_3(change, tree)
    tree.write(path)
    result = summarize_report(path)[0]
    assert result["configuration_consistent"] is (change == "capture")
    assert result["metadata_complete"] is (change == "capture")


def test_equal_function_names_in_different_modules_are_distinct_scenarios(tmp_path):
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    tree.getroot().findall(".//testcase")[1].set("classname", "tests.test_other_rag")
    tree.write(path)
    rows = summarize_report(path)
    assert len(rows) == 2
    assert {r["classname"] for r in rows} == {"tests.test_rag", "tests.test_other_rag"}
    assert all(r["runs"] == 1 and not r["mixed_pass_fail_observed"] for r in rows)


@pytest.mark.parametrize("change", CONVERSATION_SUMMARY_CHANGE_CASES)
@title("Conversation stability preserves case identity and catalog fingerprints [{param_id}]")
def test_conversation_summary_retains_case_identity_and_reviewed_catalog(tmp_path, change):
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    prepare_conversation_summary_step_3(change, tree)
    tree.write(path)
    rows = summarize_report(path)
    check_conversation_summary_outcome(change, rows)
