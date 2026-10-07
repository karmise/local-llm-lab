"""Scenario data builders and deterministic test doubles."""

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
