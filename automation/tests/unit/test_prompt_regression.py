import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.datasets.prompts import load_prompt_catalog
from llm_testkit.reporting.prompt_regression import compare_prompts
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "test_data"
CATALOG = load_prompt_catalog(DATA / "prompt-variants.json")
DATASET = load_golden_dataset(DATA / "golden-policy.json", DATA / "company-policy.txt")


def report(tmp_path):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for v in CATALOG.variants:
        row = ET.SubElement(
            suite, "testcase", classname="tests.test_prompt_regression", name=f"test_prompt[{v.id}]"
        )
        props = ET.SubElement(row, "properties")
        values = {
            "golden_case_id": "paid_leave",
            "generation_model": "model",
            "rag_iteration": "1",
            "golden_dataset_sha256": DATASET.sha256,
            "policy_sha256": DATASET.policy_sha256,
            "prompt_id": v.id,
            "prompt_version": v.version,
            "prompt_sha256": v.sha256,
            "prompt_catalog_sha256": CATALOG.sha256,
            "model_digest": "digest",
            "thinking_mode": "default",
            "workspace_configuration": json.dumps(
                {"chatModel": "model", "openAiPrompt": v.prompt, "topN": 4}
            ),
        }
        for name, value in values.items():
            ET.SubElement(props, "property", name=name, value=value)
    path = tmp_path / "runs.xml"
    ET.ElementTree(root).write(path)
    return path


def compare(path, **kwargs):
    return compare_prompts(
        path,
        catalog_path=DATA / "prompt-variants.json",
        dataset_path=DATA / "golden-policy.json",
        policy_file=DATA / "company-policy.txt",
        case_ids=["paid_leave"],
        models=["model"],
        candidate="grounded_v2",
        **kwargs,
    )


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ("none", "passed"),
        ("failed", "regression"),
        ("baseline", "baseline_failed"),
        ("missing", "incomplete"),
        ("skipped", "incomplete"),
        ("error", "incomplete"),
        ("digest", "incomplete"),
        ("prompt", "incomplete"),
        ("dataset", "incomplete"),
        ("duplicate", "incomplete"),
    ],
)
@title(
    "Prompt comparison detects regressions and rejects incomplete or uncontrolled runs [{param_id}]"
)
def test_comparison(change, expected, tmp_path):
    path = report(tmp_path)
    tree = ET.parse(path)
    suite = tree.getroot().find("testsuite")
    rows = suite.findall("testcase")
    if change in ("failed", "baseline", "skipped", "error"):
        ET.SubElement(
            rows[0] if change == "baseline" else rows[1],
            "failure" if change in ("failed", "baseline") else change,
        )
    elif change == "missing":
        suite.remove(rows[1])
    elif change == "duplicate":
        from copy import deepcopy

        duplicate = deepcopy(rows[1])
        duplicate.set("name", "other")
        suite.append(duplicate)
    elif change != "none":
        field = {
            "digest": "model_digest",
            "prompt": "prompt_sha256",
            "dataset": "golden_dataset_sha256",
        }[change]
        rows[1].find(f"./properties/property[@name='{field}']").set("value", "changed")
    tree.write(path)
    assert compare(path)["status"] == expected


@title("Prompt regression requires complete declared repetitions")
def test_missing_repetition(tmp_path):
    assert compare(report(tmp_path), repeat=2)["status"] == "incomplete"


@pytest.mark.parametrize("change", ["duplicate", "baseline", "empty", "marker"])
@title("Versioned prompt catalog rejects invalid identities and runtime data [{param_id}]")
def test_catalog_validation(tmp_path, change):
    data = json.loads((DATA / "prompt-variants.json").read_text())
    if change == "duplicate":
        data["variants"][1]["id"] = "baseline"
    elif change == "baseline":
        data["baseline"] = "absent"
    elif change == "empty":
        data["variants"][1]["prompt"] = ""
    else:
        data["variants"][1]["prompt"] += "[LLM_TESTKIT_CAPTURE:test]"
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_prompt_catalog(path)


@title(
    "Prompt baseline matches the application template and creates an opt-in case/model/prompt matrix"
)
def test_collection(framework_pytester):
    assert (
        CATALOG.variants[0].prompt
        == json.loads((ROOT.parent / "config/workspace.json").read_text())["openAiPrompt"]
    )
    shutil.copytree(DATA, framework_pytester.path / "test_data")
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.options"]')
    framework_pytester.makepyfile("""
        import pytest
        @pytest.mark.rag
        @pytest.mark.prompt_regression
        def test_prompt(prompt_variant, golden_case, generation_model, rag_iteration):
            assert golden_case.id == 'carryover_limit'
            assert prompt_variant.id in ('baseline', 'grounded_v2')
    """)
    framework_pytester.runpytest_subprocess(
        "-q", "-k", "carryover_limit", "--rag-model", "test"
    ).assert_outcomes(skipped=2, deselected=30)
    framework_pytester.runpytest_subprocess(
        "-q", "-k", "carryover_limit", "--rag-model", "test", "--run-prompt-regression"
    ).assert_outcomes(passed=2, deselected=30)


@title("Prompt comparison retains teardown errors even when a duplicate call entry failed")
def test_teardown_error_is_not_hidden(tmp_path):
    from copy import deepcopy

    path = report(tmp_path)
    tree = ET.parse(path)
    suite = tree.getroot().find("testsuite")
    row = suite.findall("testcase")[1]
    ET.SubElement(row, "error")
    duplicate = deepcopy(row)
    duplicate.remove(duplicate.find("error"))
    ET.SubElement(duplicate, "failure")
    suite.append(duplicate)
    tree.write(path)
    assert compare(path)["status"] == "incomplete"
