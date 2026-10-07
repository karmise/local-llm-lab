import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from llm_testkit.reporting.stability import summarize_report
from llm_testkit.reporting.steps import title
from test_support.builders.stability import _report

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


@pytest.mark.parametrize("same_case", [True, False], ids=["changed-dataset", "different-cases"])
@title("Golden stability preserves scenario identity and expectation fingerprints [{param_id}]")
def test_golden_summary_preserves_case_and_dataset_identity(
    tmp_path: Path, same_case: bool
) -> None:
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    for index, case in enumerate(tree.getroot().findall(".//testcase")):
        properties = case.find("properties")
        ET.SubElement(
            properties,
            "property",
            name="golden_case_id",
            value="first" if same_case or index == 0 else "second",
        )
        ET.SubElement(properties, "property", name="golden_dataset_sha256", value=str(index))
    tree.write(path)
    rows = summarize_report(path)
    if same_case:
        assert len(rows) == 1
        assert rows[0]["configuration_consistent"] is False
    else:
        assert len(rows) == 2
        assert all(row["runs"] == 1 for row in rows)
        assert all(not row["mixed_pass_fail_observed"] for row in rows)


@title("Stability summaries keep different prompt variants in separate groups")
def test_prompt_variants_are_not_reported_as_flaky_repetitions(tmp_path: Path) -> None:
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    for i, row in enumerate(tree.getroot().findall(".//testcase")):
        props = row.find("properties")
        ET.SubElement(props, "property", name="prompt_id", value=["baseline", "grounded_v2"][i])
        ET.SubElement(props, "property", name="prompt_sha256", value=str(i))
    tree.write(path)
    rows = summarize_report(path)
    assert len(rows) == 2
    assert all(not r["mixed_pass_fail_observed"] for r in rows)


@title("Counterfactual variants remain separate stability scenarios")
def test_bias_groups(tmp_path):
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    for i, row in enumerate(tree.getroot().findall(".//testcase")):
        props = row.find("properties")
        for name, value in {
            "bias_pair_id": "gender",
            "bias_variant_id": str(i + 1),
            "bias_catalog_sha256": "catalog",
        }.items():
            ET.SubElement(props, "property", name=name, value=value)
    tree.write(path)
    assert len(summarize_report(path)) == 2


@pytest.mark.parametrize("change", ["capture", "conflict", "invalid-json"])
def test_configuration_metadata_is_normalized_and_validated(tmp_path, change):
    import json

    path = _report(tmp_path, ["passed", "passed"])
    tree = ET.parse(path)
    for i, row in enumerate(tree.getroot().findall(".//testcase")):
        props = row.find("properties")
        prop = props.find("property[@name='workspace_configuration']")
        prop.set(
            "value",
            json.dumps(
                {
                    "chatModel": "qwen",
                    "openAiPrompt": "Policy\n[LLM_TESTKIT_CAPTURE:" + str(i) * 32 + "]",
                }
            ),
        )
        if change == "conflict" and i == 0:
            props.insert(0, ET.Element("property", name="model_digest", value="stale"))
        elif change == "invalid-json" and i == 0:
            prop.set("value", "not-json")
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


@pytest.mark.parametrize("change", ["different-cases", "changed-catalog", "missing-catalog"])
@title("Conversation stability preserves case identity and catalog fingerprints [{param_id}]")
def test_conversation_summary_retains_case_identity_and_reviewed_catalog(tmp_path, change):
    path = _report(tmp_path, ["passed", "failure"])
    tree = ET.parse(path)
    for i, case in enumerate(tree.getroot().findall(".//testcase")):
        props = case.find("properties")
        ET.SubElement(
            props,
            "property",
            name="conversation_case_id",
            value="greeting" if i == 0 or change != "different-cases" else "mixed_request",
        )
        if change != "missing-catalog":
            ET.SubElement(
                props,
                "property",
                name="conversation_catalog_sha256",
                value=str(i) if change == "changed-catalog" else "catalog",
            )
    tree.write(path)
    rows = summarize_report(path)
    if change == "different-cases":
        assert len(rows) == 2
        assert all(row["runs"] == 1 and not row["mixed_pass_fail_observed"] for row in rows)
    else:
        assert len(rows) == 1
        assert rows[0]["configuration_consistent"] is False
        assert rows[0]["metadata_complete"] is (change == "changed-catalog")
