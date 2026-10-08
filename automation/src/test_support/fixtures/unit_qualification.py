"""Fixtures scoped to the associated unit-test module."""

import hashlib
import json

import pytest

from llm_testkit.qualification.plan import load_plan
from test_support.data.qualification import SELECTOR


@pytest.fixture
def evidence_lab(tmp_path):
    root = tmp_path / "automation"
    (root / "tests").mkdir(parents=True)
    (root / "tests/test_example.py").write_text("def test_example():\n    assert True\n")
    (root / "src").mkdir()
    (root / "src/runtime.py").write_text("VERSION = 1\n")
    (root / "test_data").mkdir()
    (root / "test_data/policy.txt").write_text("Six days\n")
    plan = {
            "schema_version":
            1,
            "version":
            "example-v1",
            "educational_only":
            True,
            "data_sha256": {
            "policy.txt": hashlib.sha256(b"Six days\n").hexdigest()},
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
    path = root / "test_data/qualification-plan.json"
    path.write_text(json.dumps(plan))
    return root, path, load_plan(path, root)
