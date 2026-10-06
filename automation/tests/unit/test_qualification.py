import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import pytest

from llm_testkit.qualification import package
from llm_testkit.qualification.plan import expected_cells, load_plan, matching_requirements
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
SELECTOR = "tests/test_example.py::test_example"


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
        "schema_version": 1,
        "version": "example-v1",
        "educational_only": True,
        "data_sha256": {"policy.txt": hashlib.sha256(b"Six days\n").hexdigest()},
        "requirements": [
            {
                "id": "REQ-EXAMPLE",
                "phase": "OQ",
                "risk": "high",
                "description": "Example policy",
                "acceptance": "Both cases pass",
                "rationale": "A partial sample cannot demonstrate both requirements",
                "tests": [SELECTOR],
                "axes": {"golden_case_id": ["first", "second"]},
            }
        ],
    }
    path = root / "test_data/qualification-plan.json"
    path.write_text(json.dumps(plan))
    return root, path, load_plan(path, root)


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


@title("Qualification mapping resolves actual tests and requires all model and prompt cells")
def test_committed_plan_matches_real_tests():
    plan = load_plan(ROOT / "test_data/qualification-plan.json", ROOT)
    requirements = {r["id"]: r for r in plan["requirements"]}
    assert len(requirements) == 13
    assert len(expected_cells(requirements["REQ-GOLDEN"])) == 32
    assert len(expected_cells(requirements["REQ-PROMPT"])) == 64
    assert len(expected_cells(requirements["REQ-BIAS"])) == 12
    assert len([r for r in requirements.values() if r["phase"] == "IQ"]) == 3


@title("Qualification lookup can narrow a parameterized test by declared properties")
def test_matching_requirements(evidence_lab):
    _, _, plan = evidence_lab
    assert matching_requirements(plan, SELECTOR + "[first]")
    assert matching_requirements(plan, SELECTOR, {"golden_case_id": "first"})
    assert not matching_requirements(plan, SELECTOR, {"golden_case_id": "other"})
    assert not matching_requirements(plan, "tests/other.py::test_other")


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "education",
        "version",
        "requirements",
        "row",
        "id",
        "duplicate",
        "phase",
        "risk",
        "acceptance",
        "tests",
        "selector",
        "unknown",
        "escape",
        "axes",
        "axis-values",
        "axis-duplicate",
        "data",
        "data-checksum",
        "data-escape",
    ],
)
@title("Qualification plans reject malformed, stale or unsafe mappings [{param_id}]")
def test_invalid_plan(evidence_lab, change):
    root, path, _ = evidence_lab
    plan = json.loads(path.read_text())
    row = plan["requirements"][0]
    if change == "schema":
        plan["schema_version"] = True
    elif change == "education":
        plan["educational_only"] = False
    elif change == "version":
        plan["version"] = ""
    elif change == "requirements":
        plan["requirements"] = []
    elif change == "row":
        plan["requirements"] = [1]
    elif change == "id":
        row["id"] = "bad id"
    elif change == "duplicate":
        plan["requirements"].append(deepcopy(row))
    elif change in ("phase", "risk", "acceptance"):
        row[change] = ""
    elif change == "tests":
        row["tests"] = [1]
    elif change == "selector":
        row["tests"] = ["outside.py::test_example"]
    elif change == "unknown":
        row["tests"] = ["tests/test_example.py::test_unknown"]
    elif change == "escape":
        row["tests"] = ["tests/../../test_example.py::test_example"]
    elif change == "axes":
        row["axes"] = []
    elif change == "axis-values":
        row["axes"] = {"golden_case_id": []}
    elif change == "axis-duplicate":
        row["axes"] = {"golden_case_id": ["first", "first"]}
    elif change == "data":
        plan["data_sha256"] = []
    elif change == "data-checksum":
        plan["data_sha256"]["policy.txt"] = "0" * 64
    else:
        plan["data_sha256"] = {"../outside.txt": "0" * 64}
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        load_plan(path, root)


@pytest.mark.parametrize("status", ["passed", "failed", "skipped", "error", "missing"])
@title("Qualification outcomes retain failures and incomplete matrix cells [{param_id}]")
def test_trace_outcomes(evidence_lab, status):
    root, _, plan = evidence_lab
    rows = [("first", "passed", {})]
    if status != "missing":
        rows.append(("second", status, {}))
    trace = package.trace_results(plan, [write_junit(root, plan, rows)], ["OQ"])
    assert trace["status"] == {"passed": "passed", "failed": "failed"}.get(status, "incomplete")
    assert trace["review_status"] == "pending human review"
    assert trace["compliance_claim"] is False
    assert trace["requirements"][0]["cells"][1]["status"] == {
        "error": "incomplete",
        "skipped": "incomplete",
    }.get(status, status)
    if status not in ("passed", "missing"):
        assert trace["deviations"][0]["details"][0]["text"] == "original detail"


@pytest.mark.parametrize(
    "metadata",
    [
        {"qualification_plan_sha256": "stale"},
        {"test_source_sha256": "stale"},
        {"framework_source_sha256": "stale"},
        {"requirement_ids": "not-json"},
        {"requirement_ids": "null"},
        {"requirement_ids": '"REQ-EXAMPLE"'},
    ],
)
@title("Qualification evidence cannot pass with stale or malformed provenance [{param_id}]")
def test_unbound_results(evidence_lab, metadata):
    root, _, plan = evidence_lab
    junit = write_junit(root, plan, [("first", "passed", metadata), ("second", "passed", {})])
    trace = package.trace_results(plan, [junit], ["OQ"])
    assert trace["status"] == "incomplete"
    assert trace["deviations"][0]["status"] == "unbound"


@title("A later passing run does not erase a prior qualification failure")
def test_later_pass_preserves_failure(evidence_lab):
    root, _, plan = evidence_lab
    earlier = write_junit(root, plan, [("first", "failed", {})], "earlier.xml")
    later = write_junit(root, plan, [("first", "passed", {}), ("second", "passed", {})])
    assert package.trace_results(plan, [earlier, later], ["OQ"])["status"] == "failed"


@pytest.mark.parametrize("change", ["error", "metadata"])
@title(
    "Duplicate JUnit testcase entries preserve cleanup errors and conflicting metadata [{param_id}]"
)
def test_teardown_entries(evidence_lab, change):
    root, _, plan = evidence_lab
    duplicate = (
        ("first", "error", {})
        if change == "error"
        else (
            "first",
            "passed",
            {"requirement_ids": "[]"},
        )
    )
    junit = write_junit(root, plan, [("first", "passed", {}), duplicate, ("second", "passed", {})])
    trace = package.trace_results(plan, [junit], ["OQ"])
    assert trace["status"] == "incomplete"
    assert trace["deviations"]


@pytest.mark.parametrize("phases", [[], ["unknown"], ["OQ", "OQ"], ["PQ"]])
@title("Qualification packaging requires an explicit populated protocol scope [{param_id}]")
def test_phase_scope(evidence_lab, phases):
    _, _, plan = evidence_lab
    with pytest.raises(ValueError):
        package.trace_results(plan, [], phases)


@title("Legacy JUnit without requirement metadata remains incomplete evidence")
def test_legacy_results(evidence_lab):
    root, _, plan = evidence_lab
    path = root / "legacy.xml"
    path.write_text('<testsuite><testcase name="test_example"/></testsuite>')
    assert package.trace_results(plan, [path], ["OQ"])["status"] == "incomplete"


@title("Evidence packages retain exact inputs and code, verify checksums and refuse overwrite")
def test_package_integrity(evidence_lab):
    root, path, plan = evidence_lab
    junit = write_junit(root, plan, [("first", "passed", {}), ("second", "passed", {})])
    attachment = root / "reports/metrics.json"
    attachment.write_text('{"measured": 0.9}')
    output = root / "reports/package"
    args = dict(
        root=root, plan_path=path, junit_files=[junit], phases=["OQ"], attachments=[attachment]
    )
    trace = package.build_package(output, **args)
    assert trace["status"] == "passed"
    package.verify_package(output)
    assert (output / "definitions/src/runtime.py").read_bytes() == b"VERSION = 1\n"
    assert (output / "definitions/data/policy.txt").read_bytes() == b"Six days\n"
    assert (output / "junit/0-results.xml").read_bytes() == junit.read_bytes()
    assert (output / "attachments/0-metrics.json").read_bytes() == attachment.read_bytes()
    assert "REQ-EXAMPLE | OQ | high | passed" in (output / "summary.md").read_text()
    with pytest.raises(ValueError, match="already exists"):
        package.build_package(output, **args)
    (output / "summary.md").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        package.verify_package(output)


@pytest.mark.parametrize("change", ["empty", "duplicate", "outside-attachment"])
@title("Evidence packaging rejects invalid inputs and removes partial output [{param_id}]")
def test_invalid_package_inputs(evidence_lab, change):
    root, path, plan = evidence_lab
    junit = write_junit(root, plan, [])
    args = dict(root=root, plan_path=path, junit_files=[junit], phases=["OQ"])
    if change == "empty":
        args["junit_files"] = []
    elif change == "duplicate":
        args["junit_files"] = [junit, junit]
    else:
        args["attachments"] = [path]
    output = root / "reports/package"
    with pytest.raises(ValueError):
        package.build_package(output, **args)
    assert not output.exists()
    assert not list(output.parent.glob("qualification-*"))


@pytest.mark.parametrize("change", ["escape", "duplicate", "missing", "schema"])
@title(
    "Evidence verification rejects unsafe paths, duplicate entries and absent definitions [{param_id}]"
)
def test_invalid_manifest(evidence_lab, change):
    root, path, plan = evidence_lab
    output = root / "reports/package"
    package.build_package(
        output, root=root, plan_path=path, junit_files=[write_junit(root, plan, [])], phases=["OQ"]
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "escape":
        manifest["files"][0]["path"] = "../outside.json"
    elif change == "duplicate":
        manifest["files"].append(manifest["files"][0])
    elif change == "missing":
        manifest["files"] = [r for r in manifest["files"] if r["path"] != "traceability.json"]
    else:
        manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        package.verify_package(output)


@title("Qualification CLI preserves an incomplete package and returns a failing exit status")
def test_cli_incomplete(evidence_lab, monkeypatch, capsys):
    root, path, plan = evidence_lab
    output = root / "reports/package"
    junit = write_junit(root, plan, [("first", "passed", {})])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualification",
            "--root",
            str(root),
            "--plan",
            str(path),
            "--junit",
            str(junit),
            "--phase",
            "OQ",
            "--output",
            str(output),
        ],
    )
    assert package.main() == 1
    assert "incomplete" in capsys.readouterr().out
    package.verify_package(output)


@pytest.mark.parametrize("change", ["plan", "junit", "framework", "test", "data"])
@title("Evidence publication refuses input or definition changes during packaging [{param_id}]")
def test_changed_inputs(evidence_lab, monkeypatch, change):
    root, path, plan = evidence_lab
    junit = write_junit(root, plan, [("first", "passed", {}), ("second", "passed", {})])
    original_trace = package.trace_results

    def change_after_parsing(*args, **kwargs):
        trace = original_trace(*args, **kwargs)
        target = {
            "plan": path,
            "junit": junit,
            "framework": root / "src/runtime.py",
            "test": root / "tests/test_example.py",
            "data": root / "test_data/policy.txt",
        }[change]
        target.write_bytes(target.read_bytes() + b"\nchanged\n")
        return trace

    monkeypatch.setattr(package, "trace_results", change_after_parsing)
    output = root / "reports/package"
    with pytest.raises(ValueError, match="changed during packaging"):
        package.build_package(output, root=root, plan_path=path, junit_files=[junit], phases=["OQ"])
    assert not output.exists()
    assert not list(output.parent.glob("qualification-*"))


@title("Real pytest execution emits requirement, plan, source and node identity into JUnit")
def test_pytest_traceability(framework_pytester, evidence_lab):
    root, path, _ = evidence_lab
    child = framework_pytester.path
    (child / "tests").mkdir()
    (child / "tests/test_example.py").write_bytes((root / "tests/test_example.py").read_bytes())
    (child / "test_data").mkdir()
    plan = json.loads(path.read_text())
    plan["data_sha256"] = {}
    plan["requirements"][0]["axes"] = {}
    (child / "test_data/qualification-plan.json").write_text(json.dumps(plan))
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.evidence"]')
    result = framework_pytester.runpytest_subprocess("-q", "--junitxml=results.xml")
    result.assert_outcomes(passed=1)
    properties = {
        p.attrib["name"]: p.attrib["value"]
        for p in ET.parse(child / "results.xml").findall(".//property")
    }
    assert properties["test_node_id"] == SELECTOR
    assert json.loads(properties["requirement_ids"]) == ["REQ-EXAMPLE"]
    assert json.loads(properties["qualification_phases"]) == ["OQ"]
    assert (
        package.trace_results(
            load_plan(child / "test_data/qualification-plan.json", child),
            [child / "results.xml"],
            ["OQ"],
        )["status"]
        == "passed"
    )
    plan["requirements"][0]["tests"] = ["tests/test_example.py::test_unknown"]
    (child / "test_data/qualification-plan.json").write_text(json.dumps(plan))
    assert framework_pytester.runpytest_subprocess("-q").ret == pytest.ExitCode.USAGE_ERROR
