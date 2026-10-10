"""Test data builders for qualification: a small automation tree, its reviewed plan and JUnit evidence."""

import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_testkit.qualification.plan import load_plan

SELECTOR = "tests/test_example.py::test_example"
POLICY = b"Six days\n"


def reviewed_plan() -> dict[str, Any]:
    """One high-risk OQ requirement that needs both golden cases of the example test."""
    return {
            "schema_version":
            1,
            "version":
            "example-v1",
            "educational_only":
            True,
            "data_sha256": {
            "policy.txt": hashlib.sha256(POLICY).hexdigest()},
            "requirements": [{
            "id": "REQ-EXAMPLE",
            "phase": "OQ",
            "risk": "high",
            "description": "Example policy",
            "acceptance": "Both cases pass",
            "rationale": "A partial sample cannot demonstrate both requirements",
            "tests": [SELECTOR],
            "axes": {
            "golden_case_id": ["first", "second"]}}]}


def run(axis: str, status: str = "passed", **properties: str) -> tuple[str, str, dict[str, str]]:
    """One JUnit run of the example test for a golden case; ``properties`` override its recorded provenance."""
    return axis, status, properties


@dataclass
class QualificationLab:
    """An automation tree with one test, one framework source file, one data file and a reviewed plan."""

    root: Path

    @property
    def plan_path(self) -> Path:
        return self.root / "test_data/qualification-plan.json"

    def plan(self) -> dict[str, Any]:
        return load_plan(self.plan_path, self.root)

    def write_plan(self, plan: dict[str, Any]) -> None:
        self.plan_path.write_text(json.dumps(plan))

    def write_junit(self, *runs, name: str = "results.xml") -> Path:
        """JUnit for ``runs`` with provenance bound to the current plan unless a run overrides it."""
        plan = self.plan()
        suite = ET.Element("testsuite")
        for axis, status, changes in runs:
            node = f"{SELECTOR}[{axis}]"
            row = ET.SubElement(suite, "testcase", name=node)
            properties = ET.SubElement(row, "properties")
            metadata = {
                    "test_node_id": node,
                    "golden_case_id": axis,
                    "requirement_ids": json.dumps(["REQ-EXAMPLE"]),
                    "qualification_plan_sha256": plan["sha256"],
                    "test_source_sha256": plan["test_source_sha256"][SELECTOR],
                    "framework_source_sha256": plan["framework_source_sha256"],
                    **changes}
            for key, value in metadata.items():
                ET.SubElement(properties, "property", name=key, value=value)
            if status != "passed":
                outcome = ET.SubElement(row, {"failed": "failure"}.get(status, status), message="deviation")
                outcome.text = "original detail"
        path = self.root / "reports" / name
        path.parent.mkdir(exist_ok=True)
        ET.ElementTree(suite).write(path, encoding="utf-8")
        return path

    def passing_junit(self, name: str = "results.xml") -> Path:
        return self.write_junit(run("first"), run("second"), name=name)


def qualification_lab(tmp_path: Path) -> QualificationLab:
    root = tmp_path / "automation"
    (root / "tests").mkdir(parents=True)
    (root / "tests/test_example.py").write_text("def test_example():\n    assert True\n")
    (root / "src").mkdir()
    (root / "src/runtime.py").write_text("VERSION = 1\n")
    (root / "test_data").mkdir()
    (root / "test_data/policy.txt").write_bytes(POLICY)
    lab = QualificationLab(root)
    lab.write_plan(reviewed_plan())
    return lab
