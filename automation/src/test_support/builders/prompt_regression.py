"""Scenario data builders and deterministic test doubles."""

import json
import xml.etree.ElementTree as ET
from copy import deepcopy

from llm_testkit.reporting.prompt_regression import compare_prompts
from test_support.data import common as case_data
from test_support.data.prompt_regression import CATALOG as CATALOG
from test_support.data.prompt_regression import DATA as DATA
from test_support.data.prompt_regression import DATASET as DATASET
from test_support.data.prompt_regression import ROOT as ROOT


def report(tmp_path):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for v in CATALOG.variants:
        row = ET.SubElement(suite, "testcase", classname="tests.test_prompt_regression", name=f"test_prompt[{v.id}]")
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
                "model_digest": case_data.MODEL_DIGEST,
                "thinking_mode": "default",
                "workspace_configuration": json.dumps({
                "chatModel": "model",
                "openAiPrompt": v.prompt,
                "topN": 4})}
        for name, value in values.items():
            ET.SubElement(props, "property", name=name, value=value)
    path = tmp_path / "runs.xml"
    ET.ElementTree(root).write(path)
    return path


def compare(path, **kwargs):
    return compare_prompts(
            path, catalog_path=DATA / "prompt-variants.json", dataset_path=DATA / "golden-policy.json",
            policy_file=DATA / "company-policy.txt", case_ids=["paid_leave"], models=["model"], candidate="grounded_v2",
            **kwargs)


def prepare_comparison_case(change, rows, suite):
    if change in ("failed", "baseline", "skipped", "error"):
        ET.SubElement(
                rows[0] if change == "baseline" else rows[1], "failure" if change in ("failed", "baseline") else change)
    elif change == "missing":
        suite.remove(rows[1])
    elif change == "duplicate":
        pass

        duplicate = deepcopy(rows[1])
        duplicate.set("name", "other")
        suite.append(duplicate)
    elif change != "none":
        field = {"digest": "model_digest", "prompt": "prompt_sha256", "dataset": "golden_dataset_sha256"}[change]
        rows[1].find(f"./properties/property[@name='{field}']").set("value", "changed")


def prepare_catalog_validation_case(change, data):
    if change == "duplicate":
        data["variants"][1]["id"] = "baseline"
    elif change == "baseline":
        data["baseline"] = "absent"
    elif change == "empty":
        data["variants"][1]["prompt"] = ""
    else:
        data["variants"][1]["prompt"] += "[LLM_TESTKIT_CAPTURE:test]"


def prepare_conflicting_properties_within_one_testcase_case(duplicate, props, stale_first):
    if stale_first:
        props.insert(0, duplicate)
    else:
        props.append(duplicate)
