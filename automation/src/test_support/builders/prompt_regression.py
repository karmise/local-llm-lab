"""Scenario data builders and deterministic test doubles."""

import json
import xml.etree.ElementTree as ET

from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.datasets.prompts import load_prompt_catalog
from llm_testkit.reporting.prompt_regression import compare_prompts
from test_support.paths import AUTOMATION_ROOT

ROOT = AUTOMATION_ROOT


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
