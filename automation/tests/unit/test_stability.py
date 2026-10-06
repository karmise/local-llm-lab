import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from llm_testkit.reporting.stability import summarize_report
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit


def _report(tmp_path: Path, outcomes: list[str], digests: list[str] | None = None) -> Path:
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for iteration, outcome in enumerate(outcomes, 1):
        case = ET.SubElement(
            suite,
            "testcase",
            {
                "classname": "tests.test_rag",
                "name": f"test_example[qwen-run-{iteration}]",
            },
        )
        properties = ET.SubElement(case, "properties")
        values = {
            "generation_model": "qwen",
            "model_digest": digests[iteration - 1] if digests else "digest",
            "policy_sha256": "policy",
            "workspace_configuration": '{"chatModel": "qwen"}',
            "thinking_mode": "default",
        }
        for name, value in values.items():
            ET.SubElement(properties, "property", name=name, value=value)
        if outcome != "passed":
            ET.SubElement(case, outcome)
    path = tmp_path / "report.xml"
    ET.ElementTree(root).write(path)
    return path


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
