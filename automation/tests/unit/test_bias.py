import json
import shutil
import xml.etree.ElementTree as ET
from copy import deepcopy

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.datasets.bias import load_bias_cases
from llm_testkit.reporting.bias import compare_pairs
from llm_testkit.reporting.steps import title
from test_support.builders.bias import CASES, DATA, DATASET, paired_report

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("case", CASES, ids=[f"{c.pair_id}-{c.variant_id}" for c in CASES])
@title(
    "Counterfactual answers share golden facts and reject unsupported eligibility restrictions [{param_id}]"
)
def test_answers(case):
    def response(text):
        r = Response()
        r.status_code = 200
        r._content = json.dumps(
            {
                "type": "textResponse",
                "error": None,
                "close": True,
                "textResponse": text,
                "sources": [{"title": "policy", "text": (DATA / "company-policy.txt").read_text()}],
            }
        ).encode()
        return r

    assertions.assert_bias_answer(
        response(case.golden_case.reference), case=case, document_title="policy"
    )
    with pytest.raises(AssertionError, match="eligibility"):
        assertions.assert_bias_answer(
            response(case.golden_case.reference + " You are not eligible."),
            case=case,
            document_title="policy",
        )


@pytest.mark.parametrize(
    "change", ["dataset", "duplicate", "descriptor", "question", "variants", "regex"]
)
@title("Counterfactual catalog permits only the reviewed descriptor to change [{param_id}]")
def test_catalog(change, tmp_path):
    data = json.loads((DATA / "bias-policy.json").read_text())
    pair = data["pairs"][0]
    if change == "dataset":
        data["golden_dataset_sha256"] = "changed"
    elif change == "duplicate":
        data["pairs"][1]["id"] = pair["id"]
    elif change == "descriptor":
        pair["variants"][1]["descriptor"] = pair["variants"][0]["descriptor"]
    elif change == "question":
        pair["variants"][0]["question"] += " Another demand."
    elif change == "variants":
        pair["variants"].pop()
    else:
        pair["forbidden_patterns"] = {"empty": ".*"}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_bias_cases(path, DATASET)


@pytest.mark.parametrize(
    ("change", "status", "outcome"),
    [
        ("none", "passed", "passed"),
        ("one_failure", "failed", "asymmetry"),
        ("two_failures", "failed", "shared_failure"),
        ("missing", "incomplete", "incomplete"),
        ("error", "incomplete", "incomplete"),
        ("digest", "incomplete", "incomplete"),
        ("duplicate", "incomplete", "incomplete"),
    ],
)
@title(
    "Paired analysis distinguishes asymmetry, shared failures and incomplete evidence [{param_id}]"
)
def test_comparison(change, status, outcome, tmp_path):
    path = paired_report(tmp_path)
    tree = ET.parse(path)
    suite = tree.getroot().find("testsuite")
    rows = suite.findall("testcase")
    if change in ("one_failure", "two_failures"):
        ET.SubElement(rows[1], "failure")
        if change == "two_failures":
            ET.SubElement(rows[0], "failure")
    elif change == "missing":
        suite.remove(rows[1])
    elif change == "error":
        ET.SubElement(rows[1], "error")
    elif change == "digest":
        rows[1].find("./properties/property[@name='model_digest']").set("value", "changed")
    elif change == "duplicate":
        other = deepcopy(rows[1])
        other.set("name", "different")
        suite.append(other)
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
    framework_pytester.makepyfile("""
        import pytest
        @pytest.mark.bias
        @pytest.mark.rag
        def test_pair(bias_case,generation_model,rag_iteration):
            assert bias_case.pair_id == 'gender_carryover'
    """)
    framework_pytester.runpytest_subprocess(
        "-q", "--rag-model", "test", "-k", "gender"
    ).assert_outcomes(skipped=2, deselected=4)
    framework_pytester.runpytest_subprocess(
        "-q", "--rag-model", "test", "-k", "gender", "--run-bias"
    ).assert_outcomes(passed=2, deselected=4)
