"""Scenario data builders and deterministic test doubles."""

import json
import xml.etree.ElementTree as ET

from llm_testkit.datasets.bias import load_bias_cases
from llm_testkit.datasets.golden import load_golden_dataset
from test_support.paths import AUTOMATION_ROOT

DATA = AUTOMATION_ROOT / "test_data"


DATASET = load_golden_dataset(DATA / "golden-policy.json", DATA / "company-policy.txt")


CASES = load_bias_cases(DATA / "bias-policy.json", DATASET)


def paired_report(tmp_path):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for case in CASES[:2]:
        row = ET.SubElement(
            suite, "testcase", classname="tests.test_bias", name=f"test_bias[{case.variant_id}]"
        )
        props = ET.SubElement(row, "properties")
        values = {
            "bias_pair_id": case.pair_id,
            "bias_variant_id": case.variant_id,
            "generation_model": "model",
            "rag_iteration": "1",
            "bias_catalog_sha256": case.catalog_sha256,
            "golden_dataset_sha256": DATASET.sha256,
            "policy_sha256": DATASET.policy_sha256,
            "golden_case_id": case.golden_case.id,
            "model_digest": "digest",
            "thinking_mode": "default",
            "workspace_configuration": json.dumps(
                {"chatModel": "model", "openAiPrompt": "Policy", "topN": 4}
            ),
        }
        for name, value in values.items():
            ET.SubElement(props, "property", name=name, value=value)
    path = tmp_path / "results.xml"
    ET.ElementTree(root).write(path)
    return path
