"""Scenario data builders and deterministic test doubles."""

import json
import xml.etree.ElementTree as ET

from test_support.paths import AUTOMATION_ROOT

ROOT = AUTOMATION_ROOT


SELECTOR = "tests/test_example.py::test_example"


def write_junit(root, plan, rows, name="results.xml"):
    suite = ET.Element("testsuite")
    for axis, status, changes in rows:
        node = SELECTOR + f"[{axis}]"
        row = ET.SubElement(suite, "testcase", name=node)
        props = ET.SubElement(row, "properties")
        metadata = {
            "test_node_id": node,
            "golden_case_id": axis,
            "requirement_ids": json.dumps(["REQ-EXAMPLE"]),
            "qualification_plan_sha256": plan["sha256"],
            "test_source_sha256": plan["test_source_sha256"][SELECTOR],
            "framework_source_sha256": plan["framework_source_sha256"],
            **changes,
        }
        for key, value in metadata.items():
            ET.SubElement(props, "property", name=key, value=value)
        if status != "passed":
            ET.SubElement(
                row, {"failed": "failure"}.get(status, status), message="deviation"
            ).text = "original detail"
    path = root / "reports" / name
    path.parent.mkdir(exist_ok=True)
    ET.ElementTree(suite).write(path, encoding="utf-8")
    return path
