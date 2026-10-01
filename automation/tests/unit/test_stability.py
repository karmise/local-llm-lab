import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from llm_testkit import assertions
from llm_testkit.reporting.stability import summarize_report

pytestmark = pytest.mark.unit


def _report(tmp_path: Path, outcomes: list[str], digests: list[str] | None = None) -> Path:
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for iteration, outcome in enumerate(outcomes, 1):
        case = ET.SubElement(suite, "testcase", {
            "classname": "tests.test_rag", "name": f"test_example[qwen-run-{iteration}]",
        })
        properties = ET.SubElement(case, "properties")
        values = {
            "generation_model": "qwen", "model_digest": digests[iteration - 1] if digests else "digest",
            "policy_sha256": "policy", "workspace_configuration": '{"chatModel": "qwen"}',
            "thinking_mode": "default",
        }
        for name, value in values.items():
            ET.SubElement(properties, "property", name=name, value=value)
        if outcome != "passed":
            ET.SubElement(case, outcome)
    path = tmp_path / "report.xml"
    ET.ElementTree(root).write(path)
    return path


def test_summary_preserves_mixed_pass_fail_results(tmp_path: Path) -> None:
    row = summarize_report(_report(tmp_path, ["passed", "failure", "passed"]))[0]
    assertions.assert_field_equals(row, "runs", 3)
    assertions.assert_field_equals(row, "passed", 2)
    assertions.assert_field_equals(row, "failed", 1)
    assertions.assert_field_equals(row, "mixed_pass_fail_observed", True)
    assertions.assert_field_equals(row, "configuration_consistent", True)


def test_summary_detects_configuration_changes(tmp_path: Path) -> None:
    row = summarize_report(_report(tmp_path, ["passed", "failure"], ["first", "second"]))[0]
    assertions.assert_field_equals(row, "configuration_consistent", False)


def test_setup_error_without_metadata_is_not_a_model_failure(tmp_path: Path) -> None:
    path = _report(tmp_path, ["passed", "error"])
    tree = ET.parse(path)
    failed_case = tree.getroot().findall(".//testcase")[1]
    failed_case.remove(failed_case.find("properties"))
    tree.write(path)
    row = summarize_report(path)[0]
    assertions.assert_field_equals(row, "runs", 2)
    assertions.assert_field_equals(row, "errored", 1)
    assertions.assert_field_equals(row, "failed", 0)
    assertions.assert_field_equals(row, "mixed_pass_fail_observed", False)
    assertions.assert_field_equals(row, "metadata_complete", False)


def test_duplicate_call_and_teardown_entries_count_as_one_run(tmp_path: Path) -> None:
    path = _report(tmp_path, ["failure"])
    tree = ET.parse(path)
    suite = tree.getroot().find("testsuite")
    duplicate = ET.SubElement(suite, "testcase", {
        "classname": "tests.test_rag", "name": "test_example[qwen-run-1]",
    })
    ET.SubElement(duplicate, "error")
    tree.write(path)
    row = summarize_report(path)[0]
    assertions.assert_field_equals(row, "runs", 1)
    assertions.assert_field_equals(row, "failed", 1)
    assertions.assert_field_equals(row, "errored", 1)
