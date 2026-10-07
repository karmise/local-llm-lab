import json
import shutil
import sys
import xml.etree.ElementTree as ET

import pytest

from llm_testkit.qualification import package
from llm_testkit.qualification.plan import expected_cells, load_plan, matching_requirements
from llm_testkit.reporting.steps import title
from test_support.builders.qualification import (
    check_trace_outcomes_outcome,
    make_change_after_parsing_stub,
    make_teardown_entries,
    prepare_conflicting_inline_provenance_cannot_pass_case,
    prepare_invalid_manifest_case,
    prepare_invalid_package_inputs_case,
    prepare_invalid_plan_case,
    prepare_package_verification_binds_outcomes_to_inputs_case,
    prepare_plan_rejects_ambiguous_paths_and_phase_types_case,
    prepare_trace_outcomes_case,
    write_junit,
)
from test_support.data.qualification import (
    CHANGED_INPUTS_CHANGE_CASES,
    CONFLICTING_INLINE_PROVENANCE_CANNOT_PASS_STALE_FIRST_CASES,
    INVALID_MANIFEST_CHANGE_CASES,
    INVALID_PACKAGE_INPUTS_CHANGE_CASES,
    INVALID_PLAN_CHANGE_CASES,
    PACKAGE_VERIFICATION_BINDS_OUTCOMES_TO_INPUTS_CHANGE_CASES,
    PHASE_SCOPE_PHASES_CASES,
    PLAN_REJECTS_AMBIGUOUS_PATHS_AND_PHASE_TYPES_CHANGE_CASES,
    ROOT,
    SELECTOR,
    TEARDOWN_ENTRIES_CHANGE_CASES,
    TRACE_OUTCOMES_STATUS_CASES,
    UNBOUND_RESULTS_METADATA_CASES,
)
from test_support.fixtures.unit_qualification import evidence_lab as evidence_lab

pytestmark = pytest.mark.unit


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
    INVALID_PLAN_CHANGE_CASES,
)
@title("Qualification plans reject malformed, stale or unsafe mappings [{param_id}]")
def test_invalid_plan(evidence_lab, change):
    root, path, _ = evidence_lab
    plan = json.loads(path.read_text())
    row = plan["requirements"][0]
    prepare_invalid_plan_case(change, plan, row)
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        load_plan(path, root)


@pytest.mark.parametrize("status", TRACE_OUTCOMES_STATUS_CASES)
@title("Qualification outcomes retain failures and incomplete matrix cells [{param_id}]")
def test_trace_outcomes(evidence_lab, status):
    root, _, plan = evidence_lab
    rows = [("first", "passed", {})]
    prepare_trace_outcomes_case(rows, status)
    trace = package.trace_results(plan, [write_junit(root, plan, rows)], ["OQ"])
    assert trace["status"] == {"passed": "passed", "failed": "failed"}.get(status, "incomplete")
    assert trace["review_status"] == "pending human review"
    assert trace["compliance_claim"] is False
    assert trace["requirements"][0]["cells"][1]["status"] == {
        "error": "incomplete",
        "skipped": "incomplete",
    }.get(status, status)
    check_trace_outcomes_outcome(status, trace)


@pytest.mark.parametrize(
    "metadata",
    UNBOUND_RESULTS_METADATA_CASES,
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


@pytest.mark.parametrize("change", TEARDOWN_ENTRIES_CHANGE_CASES)
@title(
    "Duplicate JUnit testcase entries preserve cleanup errors and conflicting metadata [{param_id}]"
)
def test_teardown_entries(evidence_lab, change):
    root, _, plan = evidence_lab
    duplicate = make_teardown_entries(change)
    junit = write_junit(root, plan, [("first", "passed", {}), duplicate, ("second", "passed", {})])
    trace = package.trace_results(plan, [junit], ["OQ"])
    assert trace["status"] == "incomplete"
    assert trace["deviations"]


@pytest.mark.parametrize("phases", PHASE_SCOPE_PHASES_CASES)
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


@pytest.mark.parametrize("change", INVALID_PACKAGE_INPUTS_CHANGE_CASES)
@title("Evidence packaging rejects invalid inputs and removes partial output [{param_id}]")
def test_invalid_package_inputs(evidence_lab, change):
    root, path, plan = evidence_lab
    junit = write_junit(root, plan, [])
    args = dict(root=root, plan_path=path, junit_files=[junit], phases=["OQ"])
    prepare_invalid_package_inputs_case(args, change, junit, path)
    output = root / "reports/package"
    with pytest.raises(ValueError):
        package.build_package(output, **args)
    assert not output.exists()
    assert not list(output.parent.glob("qualification-*"))


@pytest.mark.parametrize("change", INVALID_MANIFEST_CHANGE_CASES)
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
    prepare_invalid_manifest_case(change, manifest)
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


@pytest.mark.parametrize("change", CHANGED_INPUTS_CHANGE_CASES)
@title("Evidence publication refuses input or definition changes during packaging [{param_id}]")
def test_changed_inputs(evidence_lab, monkeypatch, change):
    root, path, plan = evidence_lab
    junit = write_junit(root, plan, [("first", "passed", {}), ("second", "passed", {})])
    original_trace = package.trace_results

    change_after_parsing = make_change_after_parsing_stub(change, junit, original_trace, path, root)

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


@pytest.mark.parametrize("stale_first", CONFLICTING_INLINE_PROVENANCE_CANNOT_PASS_STALE_FIRST_CASES)
def test_conflicting_inline_provenance_cannot_pass(evidence_lab, stale_first):
    root, _, plan = evidence_lab
    junit = write_junit(root, plan, [("first", "passed", {}), ("second", "passed", {})])
    tree = ET.parse(junit)
    props = tree.getroot().find(".//properties")
    duplicate = ET.Element("property", name="qualification_plan_sha256", value="stale")
    prepare_conflicting_inline_provenance_cannot_pass_case(duplicate, props, stale_first)
    tree.write(junit)
    assert package.trace_results(plan, [junit], ["OQ"])["status"] == "incomplete"


@pytest.mark.parametrize("change", PACKAGE_VERIFICATION_BINDS_OUTCOMES_TO_INPUTS_CHANGE_CASES)
def test_package_verification_binds_outcomes_to_inputs(evidence_lab, change):
    root, path, plan = evidence_lab
    output = root / "reports/package"
    package.build_package(
        output, root=root, plan_path=path, junit_files=[write_junit(root, plan, [])], phases=["OQ"]
    )
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    prepare_package_verification_binds_outcomes_to_inputs_case(change, manifest, output)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        package.verify_package(output)


def test_nested_fixture_changes_invalidate_framework_provenance(evidence_lab):
    root, path, original = evidence_lab
    (root / "tests/ui").mkdir()
    (root / "tests/ui/conftest.py").write_text("FIXTURE_VERSION = 1\n")
    changed = load_plan(path, root)
    assert changed["framework_source_sha256"] != original["framework_source_sha256"]


@pytest.mark.parametrize("change", PLAN_REJECTS_AMBIGUOUS_PATHS_AND_PHASE_TYPES_CHANGE_CASES)
def test_plan_rejects_ambiguous_paths_and_phase_types(evidence_lab, change):
    root, path, _ = evidence_lab
    data = json.loads(path.read_text())
    prepare_plan_rejects_ambiguous_paths_and_phase_types_case(change, data)
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_plan(path, root)


def test_package_verification_is_independent_of_original_directory(evidence_lab):

    root, path, plan = evidence_lab
    output = root / "reports/package"
    package.build_package(
        output,
        root=root,
        plan_path=path,
        junit_files=[write_junit(root, plan, [("first", "passed", {}), ("second", "passed", {})])],
        phases=["OQ"],
    )
    moved = root.parent / "archived-evidence"
    shutil.move(output, moved)
    shutil.rmtree(root)
    package.verify_package(moved)
