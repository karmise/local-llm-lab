import json
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy

import pytest

from llm_testkit.datasets.prompts import load_prompt_catalog
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import pytest_runs
from test_support.assertions import values as value_checks
from test_support.builders.prompt_regression import (
    compare,
    prepare_catalog_validation_case,
    prepare_comparison_case,
    prepare_conflicting_properties_within_one_testcase_case,
    report,
)
from test_support.data.prompt_regression import (
    CATALOG,
    CATALOG_VALIDATION_CHANGE_CASES,
    COMPARISON_CHANGE_EXPECTED_CASES,
    CONFLICTING_PROPERTIES_WITHIN_ONE_TESTCASE_STALE_FIRST_CASES,
    DATA,
    ROOT,
)
from test_support.data.scripts.prompt_regression import COLLECTION_MAKEPYFILE_SOURCE

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("change", "expected"),
    COMPARISON_CHANGE_EXPECTED_CASES,
)
@title(
    "Prompt comparison detects regressions and rejects incomplete or uncontrolled runs [{param_id}]"
)
def test_comparison(change, expected, tmp_path):
    path = report(tmp_path)
    tree = ET.parse(path)
    suite = tree.getroot().find("testsuite")
    rows = suite.findall("testcase")
    prepare_comparison_case(change, rows, suite)
    tree.write(path)
    value_checks.equal(compare(path)["status"], expected)


@title("Prompt regression requires complete declared repetitions")
def test_missing_repetition(tmp_path):
    value_checks.equal(compare(report(tmp_path), repeat=2)["status"], "incomplete")


@pytest.mark.parametrize("change", CATALOG_VALIDATION_CHANGE_CASES)
@title("Versioned prompt catalog rejects invalid identities and runtime data [{param_id}]")
def test_catalog_validation(tmp_path, change):
    data = json.loads((DATA / "prompt-variants.json").read_text())
    prepare_catalog_validation_case(change, data)
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(data))
    errors.rejects(lambda: load_prompt_catalog(path), expected=ValueError)


@title(
    "Prompt baseline matches the application template and creates an opt-in case/model/prompt matrix"
)
def test_collection(framework_pytester):
    value_checks.equal(
        CATALOG.variants[0].prompt,
        json.loads((ROOT.parent / "config/workspace.json").read_text())["openAiPrompt"],
    )
    shutil.copytree(DATA, framework_pytester.path / "test_data")
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    framework_pytester.makepyfile(COLLECTION_MAKEPYFILE_SOURCE)
    pytest_runs.outcomes(
        framework_pytester.runpytest_subprocess(
            "-q", "-k", "carryover_limit", "--rag-model", "test"
        ),
        skipped=2,
        deselected=30,
    )
    pytest_runs.outcomes(
        framework_pytester.runpytest_subprocess(
            "-q", "-k", "carryover_limit", "--rag-model", "test", "--run-prompt-regression"
        ),
        passed=2,
        deselected=30,
    )


@title("Prompt comparison retains teardown errors even when a duplicate call entry failed")
def test_teardown_error_is_not_hidden(tmp_path, xml_property_factory):

    path = report(tmp_path)
    tree = ET.parse(path)
    suite = tree.getroot().find("testsuite")
    row = suite.findall("testcase")[1]
    xml_property_factory(row, "error")
    duplicate = deepcopy(row)
    duplicate.remove(duplicate.find("error"))
    xml_property_factory(duplicate, "failure")
    suite.append(duplicate)
    tree.write(path)
    value_checks.equal(compare(path)["status"], "incomplete")


@pytest.mark.parametrize(
    "stale_first", CONFLICTING_PROPERTIES_WITHIN_ONE_TESTCASE_STALE_FIRST_CASES
)
def test_conflicting_properties_within_one_testcase(tmp_path, stale_first, xml_element_factory):
    path = report(tmp_path)
    tree = ET.parse(path)
    props = tree.getroot().find(".//properties")
    duplicate = xml_element_factory("property", name="policy_sha256", value="stale")
    prepare_conflicting_properties_within_one_testcase_case(duplicate, props, stale_first)
    tree.write(path)
    result = compare(path)
    value_checks.equal(result["status"], "incomplete")
    value_checks.truthy(result["errors"])
