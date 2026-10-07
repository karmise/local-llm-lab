"""Scenario data builders and deterministic test doubles."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path


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


def prepare_configuration_metadata_case(change, i, prop, props):
    if change == "conflict" and i == 0:
        props.insert(0, ET.Element("property", name="model_digest", value="stale"))
    elif change == "invalid-json" and i == 0:
        prop.set("value", "not-json")


def prepare_conversation_summary_case(change, i, props):
    if change != "missing-catalog":
        ET.SubElement(
            props,
            "property",
            name="conversation_catalog_sha256",
            value=str(i) if change == "changed-catalog" else "catalog",
        )


def add_golden_case_metadata(same_case, tree):
    for index, case in enumerate(tree.getroot().findall(".//testcase")):
        properties = case.find("properties")
        ET.SubElement(
            properties,
            "property",
            name="golden_case_id",
            value="first" if same_case or index == 0 else "second",
        )
        ET.SubElement(properties, "property", name="golden_dataset_sha256", value=str(index))


def add_prompt_variant_metadata(tree):
    for i, row in enumerate(tree.getroot().findall(".//testcase")):
        props = row.find("properties")
        ET.SubElement(props, "property", name="prompt_id", value=["baseline", "grounded_v2"][i])
        ET.SubElement(props, "property", name="prompt_sha256", value=str(i))


def add_bias_group_metadata(tree):
    for i, row in enumerate(tree.getroot().findall(".//testcase")):
        props = row.find("properties")
        for name, value in {
            "bias_pair_id": "gender",
            "bias_variant_id": str(i + 1),
            "bias_catalog_sha256": "catalog",
        }.items():
            ET.SubElement(props, "property", name=name, value=value)


def add_configuration_metadata(change, tree):
    for i, row in enumerate(tree.getroot().findall(".//testcase")):
        props = row.find("properties")
        prop = props.find("property[@name='workspace_configuration']")
        prop.set(
            "value",
            json.dumps(
                {
                    "chatModel": "qwen",
                    "openAiPrompt": "Policy\n[LLM_TESTKIT_CAPTURE:" + str(i) * 32 + "]",
                }
            ),
        )
        prepare_configuration_metadata_case(change, i, prop, props)


def add_conversation_metadata(change, tree):
    for i, case in enumerate(tree.getroot().findall(".//testcase")):
        props = case.find("properties")
        ET.SubElement(
            props,
            "property",
            name="conversation_case_id",
            value="greeting" if i == 0 or change != "different-cases" else "mixed_request",
        )
        prepare_conversation_summary_case(change, i, props)
