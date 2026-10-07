"""Scenario data builders and deterministic test doubles."""

import json
import xml.etree.ElementTree as ET
from copy import deepcopy

from requests import Response

from test_support.data import common as case_data
from test_support.data.bias import CASES as CASES
from test_support.data.bias import DATA as DATA
from test_support.data.bias import DATASET as DATASET


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
            "model_digest": case_data.MODEL_DIGEST,
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


def make_response_stub():
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

    return response


def prepare_catalog_case(change, data, pair):
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


def prepare_comparison_case(change, rows, suite):
    if change in ("one_failure", "two_failures"):
        ET.SubElement(rows[1], "failure")
        if change == "two_failures":
            ET.SubElement(rows[0], "failure")
    elif change == "missing":
        suite.remove(rows[1])
    elif change == "error":
        ET.SubElement(rows[1], "error")
    elif change == case_data.MODEL_DIGEST:
        rows[1].find("./properties/property[@name='model_digest']").set("value", "changed")
    elif change == "duplicate":
        other = deepcopy(rows[1])
        other.set("name", "different")
        suite.append(other)
