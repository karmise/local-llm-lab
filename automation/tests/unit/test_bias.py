import json
import shutil
import xml.etree.ElementTree as ET

import pytest

from llm_testkit import assertions
from llm_testkit.datasets.bias import load_bias_cases
from llm_testkit.reporting.bias import compare_pairs
from llm_testkit.reporting.steps import title
from test_support.builders.bias import (
    make_response_stub,
    paired_report,
    prepare_catalog_case,
    prepare_comparison_case,
)
from test_support.data.bias import (
    ANSWERS_CASE_CASES,
    ANSWERS_CASE_IDS,
    CATALOG_CHANGE_CASES,
    COMPARISON_CHANGE_STATUS_OUTCOME_CASES,
    DATA,
    DATASET,
)
from test_support.data.scripts.bias import SELECTION_MAKEPYFILE_SOURCE

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("case", ANSWERS_CASE_CASES, ids=ANSWERS_CASE_IDS)
@title(
    "Counterfactual answers share golden facts and reject unsupported eligibility restrictions [{param_id}]"
)
def test_answers(case):
    response = make_response_stub()

    assertions.assert_bias_answer(
        response(case.golden_case.reference), case=case, document_title="policy"
    )
    with pytest.raises(AssertionError, match="eligibility"):
        assertions.assert_bias_answer(
            response(case.golden_case.reference + " You are not eligible."),
            case=case,
            document_title="policy",
        )


@pytest.mark.parametrize("change", CATALOG_CHANGE_CASES)
@title("Counterfactual catalog permits only the reviewed descriptor to change [{param_id}]")
def test_catalog(change, tmp_path):
    data = json.loads((DATA / "bias-policy.json").read_text())
    pair = data["pairs"][0]
    prepare_catalog_case(change, data, pair)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_bias_cases(path, DATASET)


@pytest.mark.parametrize(
    ("change", "status", "outcome"),
    COMPARISON_CHANGE_STATUS_OUTCOME_CASES,
)
@title(
    "Paired analysis distinguishes asymmetry, shared failures and incomplete evidence [{param_id}]"
)
def test_comparison(change, status, outcome, tmp_path):
    path = paired_report(tmp_path)
    tree = ET.parse(path)
    suite = tree.getroot().find("testsuite")
    rows = suite.findall("testcase")
    prepare_comparison_case(change, rows, suite)
    tree.write(path)
    report = compare_pairs(
        path,
        catalog=DATA / "bias-policy.json",
        dataset_path=DATA / "golden-policy.json",
        policy=DATA / "company-policy.txt",
        pair_ids=["gender_carryover"],
        models=["model"],
    )
    assert report["status"] == status
    assert report["comparisons"][0]["outcome"] == outcome


@title("Bias variants collect as independent opt-in case/model runs")
def test_selection(framework_pytester):
    shutil.copytree(DATA, framework_pytester.path / "test_data")
    framework_pytester.makeconftest('pytest_plugins=["llm_testkit.pytest_support.options"]')
    framework_pytester.makepyfile(SELECTION_MAKEPYFILE_SOURCE)
    framework_pytester.runpytest_subprocess(
        "-q", "--rag-model", "test", "-k", "gender"
    ).assert_outcomes(skipped=2, deselected=4)
    framework_pytester.runpytest_subprocess(
        "-q", "--rag-model", "test", "-k", "gender", "--run-bias"
    ).assert_outcomes(passed=2, deselected=4)
