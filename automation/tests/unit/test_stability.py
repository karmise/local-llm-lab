import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from llm_testkit.reporting.stability import summarize_report
from llm_testkit.reporting.steps import title
from test_support.assertions import values as value_checks
from test_support.assertions.stability import (
    check_conversation_summary_outcome,
    check_golden_summary_outcome,
)
from test_support.builders.stability import (
    _report,
    add_bias_group_metadata,
    add_configuration_metadata,
    add_conversation_metadata,
    add_golden_case_metadata,
    add_prompt_variant_metadata,
)
from test_support.data import common as case_data
from test_support.data.stability import (
    CLASSNAME_NAME_TEST_EXAMPLE_QWEN_RUN_1_INPUT,
    CONFIGURATION_METADATA_CHANGE_CASES,
    CONVERSATION_SUMMARY_CHANGE_CASES,
    GOLDEN_SUMMARY_SAME_CASE_CASES,
    GOLDEN_SUMMARY_SAME_CASE_IDS,
)

pytestmark = pytest.mark.unit


@title("Stability summary preserves mixed passed and failed runs")
def test_summary_preserves_mixed_pass_fail_results(tmp_path: Path) -> None:
    row = summarize_report(_report(tmp_path, ["passed", "failure", "passed"]))[0]
    value_checks.equal(row["runs"], 3)
    value_checks.equal(row["passed"], 2)
    value_checks.equal(row["failed"], 1)
    value_checks.identical(row["mixed_pass_fail_observed"], True)
    value_checks.identical(row["configuration_consistent"], True)


@title("Stability summary detects changes in model configuration")
def test_summary_detects_configuration_changes(tmp_path: Path) -> None:
    row = summarize_report(_report(tmp_path, ["passed", "failure"], ["first", "second"]))[0]
    value_checks.identical(row["configuration_consistent"], False)


@title("Stability summary distinguishes setup errors from model failures")
def test_setup_error_without_metadata_is_not_a_model_failure(tmp_path: Path) -> None:
    path = _report(tmp_path, ["passed", "error"])
    tree = ET.parse(path)
    failed_case = tree.getroot().findall(".//testcase")[1]
    failed_case.remove(failed_case.find("properties"))
    tree.write(path)
    row = summarize_report(path)[0]
    value_checks.equal(row["runs"], 2)
    value_checks.equal(row["errored"], 1)
    value_checks.equal(row["failed"], 0)
    value_checks.identical(row["mixed_pass_fail_observed"], False)
    value_checks.identical(row["metadata_complete"], False)


@title("Stability summary counts call and teardown entries as one run")
def test_duplicate_call_and_teardown_entries_count_as_one_run(
        tmp_path: Path, xml_property_factory) -> None:  # fmt: skip
    path = _report(tmp_path, ["failure"])
    tree = ET.parse(path)
    suite = tree.getroot().find("testsuite")
    duplicate = xml_property_factory(
        suite,
        "testcase",
        case_data.fresh(CLASSNAME_NAME_TEST_EXAMPLE_QWEN_RUN_1_INPUT),
    )
    xml_property_factory(duplicate, "error")
    tree.write(path)
    row = summarize_report(path)[0]
    value_checks.equal(row["runs"], 1)
    value_checks.equal(row["failed"], 1)
    value_checks.equal(row["errored"], 1)


@pytest.mark.parametrize(
    "same_case",
    GOLDEN_SUMMARY_SAME_CASE_CASES,
    ids=GOLDEN_SUMMARY_SAME_CASE_IDS,
)
@title("Golden stability preserves scenario identity and expectation fingerprints [{param_id}]")
def test_golden_summary_preserves_case_and_dataset_identity(tmp_path: Path,
                                                            same_case: bool) -> None:  # fmt: skip
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    add_golden_case_metadata(same_case, tree)
    tree.write(path)
    rows = summarize_report(path)
    check_golden_summary_outcome(rows, same_case)


@title("Stability summaries keep different prompt variants in separate groups")
def test_prompt_variants_are_not_reported_as_flaky_repetitions(tmp_path: Path) -> None:
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    add_prompt_variant_metadata(tree)
    tree.write(path)
    rows = summarize_report(path)
    value_checks.length(rows, 2)
    value_checks.all_true((not r["mixed_pass_fail_observed"] for r in rows))


@title("Counterfactual variants remain separate stability scenarios")
def test_bias_groups(tmp_path):
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    add_bias_group_metadata(tree)
    tree.write(path)
    value_checks.length(summarize_report(path), 2)


@pytest.mark.parametrize("change", CONFIGURATION_METADATA_CHANGE_CASES)
def test_configuration_metadata_is_normalized_and_validated(tmp_path, change):

    path = _report(tmp_path, ["passed", "passed"])
    tree = ET.parse(path)
    add_configuration_metadata(change, tree)
    tree.write(path)
    result = summarize_report(path)[0]
    value_checks.identical(result["configuration_consistent"], change == "capture")
    value_checks.identical(result["metadata_complete"], change == "capture")


def test_equal_function_names_in_different_modules_are_distinct_scenarios(tmp_path):
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    tree.getroot().findall(".//testcase")[1].set("classname", "tests.test_other_rag")
    tree.write(path)
    rows = summarize_report(path)
    value_checks.length(rows, 2)
    value_checks.equal({r["classname"] for r in rows}, {"tests.test_rag", "tests.test_other_rag"})
    value_checks.all_true((r["runs"] == 1 and (not r["mixed_pass_fail_observed"]) for r in rows))


@pytest.mark.parametrize("change", CONVERSATION_SUMMARY_CHANGE_CASES)
@title("Conversation stability preserves case identity and catalog fingerprints [{param_id}]")
def test_conversation_summary_retains_case_identity_and_reviewed_catalog(tmp_path, change):
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    add_conversation_metadata(change, tree)
    tree.write(path)
    rows = summarize_report(path)
    check_conversation_summary_outcome(change, rows)
