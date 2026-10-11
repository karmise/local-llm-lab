"""Qualification package: traced outcomes and an exact, verifiable evidence package that keeps every deviation."""

import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET

import pytest

from llm_testkit.qualification import package
from llm_testkit.qualification.package import build_package, trace_results, verify_package
from llm_testkit.qualification.plan import load_plan
from llm_testkit.reporting.steps import title
from test_support.builders.qualification import SELECTOR, qualification_lab, reviewed_plan, run

pytestmark = pytest.mark.unit


@pytest.fixture
def lab(tmp_path):
    return qualification_lab(tmp_path)


def trace(lab, *junit, phases=("OQ", )):
    return trace_results(lab.plan(), list(junit), list(phases))


@title("Bound passing runs of every cell pass, and the trace never claims compliance")
def test_trace_passes(lab):
    junit = lab.passing_junit()

    result = trace(lab, junit)

    assert {
            k: v
            for k, v in result.items() if k not in ("requirements", "interpretation")} == {
            "schema_version": 1,
            "educational_only": True,
            "input_sha256": {
            str(junit.resolve()): hashlib.sha256(junit.read_bytes()).hexdigest()},
            "protocol_scope": ["OQ"],
            "status": "passed",
            "plan_version": "example-v1",
            "plan_sha256": lab.plan()["sha256"],
            "deviations": [],
            "review_status": "pending human review",
            "compliance_claim": False}
    assert result["requirements"] == [{
            "id":
            "REQ-EXAMPLE",
            "phase":
            "OQ",
            "risk":
            "high",
            "acceptance":
            "Both cases pass",
            "status":
            "passed",
            "cells": [{
            "test": SELECTOR,
            "axes": {
            "golden_case_id": axis},
            "status": "passed",
            "observed_outcomes": ["passed"]} for axis in ("first", "second")]}]
    assert "no electronic signature, formal approval or GxP certification" in result["interpretation"]


@pytest.mark.parametrize(("status", "cell", "overall"), [
        pytest.param("failed", "failed", "failed", id="failed"),
        pytest.param("skipped", "incomplete", "incomplete", id="skipped"),
        pytest.param("error", "incomplete", "incomplete", id="error")])
@title("A failed, skipped or errored cell is kept as a deviation with its original detail [{param_id}]")
def test_trace_keeps_deviations(lab, status, cell, overall):
    result = trace(lab, lab.write_junit(run("first"), run("second", status)))

    assert (result["status"], result["requirements"][0]["status"]) == (overall, overall)
    assert result["requirements"][0]["cells"][1]["status"] == cell
    deviation, = result["deviations"]
    assert (deviation["requirement"], deviation["node"],
            deviation["status"]) == ("REQ-EXAMPLE", f"{SELECTOR}[second]", status)
    assert deviation["details"] == [{"kind": status, "message": "deviation", "text": "original detail"}]


@title("A cell without any run is missing and leaves the trace incomplete")
def test_trace_reports_missing_cell(lab):
    result = trace(lab, lab.write_junit(run("first")))

    assert result["status"] == "incomplete"
    assert result["requirements"][0]["cells"][1] == {
            "test": SELECTOR,
            "axes": {
            "golden_case_id": "second"},
            "status": "missing",
            "observed_outcomes": []}


@pytest.mark.parametrize(
        "provenance", [
        pytest.param({"qualification_plan_sha256": "stale"}, id="stale-plan"),
        pytest.param({"test_source_sha256": "stale"}, id="stale-test"),
        pytest.param({"framework_source_sha256": "stale"}, id="stale-framework"),
        pytest.param({"requirement_ids": "not-json"}, id="invalid-json"),
        pytest.param({"requirement_ids": "null"}, id="null-requirements"),
        pytest.param({"requirement_ids": '"REQ-EXAMPLE"'}, id="string-requirements"),
        pytest.param({"requirement_ids": '["REQ-OTHER"]'}, id="other-requirement")])
@title("A run with stale or malformed provenance is unbound and cannot pass [{param_id}]")
def test_trace_rejects_unbound_runs(lab, provenance):
    result = trace(lab, lab.write_junit(run("first", **provenance), run("second")))

    assert result["status"] == "incomplete"
    assert [(d["node"], d["status"]) for d in result["deviations"]] == [(f"{SELECTOR}[first]", "unbound")]


@title("A run without a requirement list is unbound")
def test_trace_rejects_run_without_requirements(lab):
    junit = lab.write_junit(run("first"), run("second"))
    tree = ET.parse(junit)
    properties = tree.getroot().find(".//properties")
    properties.remove(properties.find("property[@name='requirement_ids']"))
    tree.write(junit)

    assert trace(lab, junit)["deviations"][0]["status"] == "unbound"


@title("A later passing run does not erase an earlier failure")
def test_later_pass_preserves_failure(lab):
    earlier = lab.write_junit(run("first", "failed"), name="earlier.xml")
    later = lab.passing_junit()

    result = trace(lab, earlier, later)

    assert result["status"] == "failed"
    assert result["requirements"][0]["cells"][0]["observed_outcomes"] == ["failed", "passed"]


@title("An unbound run next to a passing one leaves the cell incomplete")
def test_unbound_run_next_to_pass_is_incomplete(lab):
    stale = lab.write_junit(run("first", qualification_plan_sha256="stale"), name="stale.xml")

    result = trace(lab, stale, lab.passing_junit())

    assert result["requirements"][0]["cells"][0]["status"] == "incomplete"


@pytest.mark.parametrize(
        "teardown", [
        pytest.param(run("first", "error"), id="teardown-error"),
        pytest.param(run("first", requirement_ids="[]"), id="conflicting-metadata")])
@title("A teardown entry of the same test keeps its error or conflicting metadata [{param_id}]")
def test_trace_keeps_teardown_entries(lab, teardown):
    result = trace(lab, lab.write_junit(run("first"), teardown, run("second")))

    assert result["status"] == "incomplete"
    assert result["deviations"][0]["node"] == f"{SELECTOR}[first]"


@title("Conflicting metadata is named in the deviation details")
def test_trace_names_conflicting_metadata(lab):
    result = trace(
            lab,
            lab.write_junit(run("first"), run("first", golden_case_id="first", requirement_ids="[]"), run("second")))

    assert result["deviations"][0]["details"] == ["Conflicting metadata: requirement_ids"]


@pytest.mark.parametrize("stale_first", [pytest.param(True, id="stale-first"), pytest.param(False, id="stale-last")])
@title("A repeated provenance property with a stale value cannot pass [{param_id}]")
def test_conflicting_inline_provenance_cannot_pass(lab, stale_first):
    junit = lab.passing_junit()
    tree = ET.parse(junit)
    properties = tree.getroot().find(".//properties")
    stale = ET.Element("property", name="qualification_plan_sha256", value="stale")
    properties.insert(0 if stale_first else len(properties), stale)
    tree.write(junit)

    assert trace(lab, junit)["status"] == "incomplete"


@title("JUnit without requirement metadata is ignored and leaves the trace incomplete")
def test_legacy_results_are_incomplete(lab):
    path = lab.root / "legacy.xml"
    path.write_text('<testsuite><testcase name="test_example"/></testsuite>')

    result = trace(lab, path)

    assert (result["status"], result["deviations"]) == ("incomplete", [])


@pytest.mark.parametrize(
        "phases", [
        pytest.param([], id="none"),
        pytest.param(["unknown"], id="unknown"),
        pytest.param(["OQ", "OQ"], id="duplicate")])
@title("Packaging requires distinct, known protocol phases [{param_id}]")
def test_trace_rejects_invalid_phases(lab, phases):
    with pytest.raises(ValueError, match="Select distinct IQ/OQ/PQ protocol phases"):
        trace(lab, phases=phases)


@title("All three protocol phases can be selected; runs bind only to the requirements they name")
def test_trace_accepts_all_phases(lab):
    plan = reviewed_plan()
    base = plan["requirements"][0]
    plan["requirements"] += [base | {"id": "REQ-IQ", "phase": "IQ"}, base | {"id": "REQ-PQ", "phase": "PQ"}]
    lab.write_plan(plan)

    result = trace(lab, lab.passing_junit(), phases=("IQ", "OQ", "PQ"))

    assert result["protocol_scope"] == ["IQ", "OQ", "PQ"]
    assert [(r["id"], r["status"]) for r in result["requirements"]] == [("REQ-EXAMPLE", "passed"),
            ("REQ-IQ", "incomplete"), ("REQ-PQ", "incomplete")]


@title("Entries of one node recorded under different testcase names are merged by node id")
def test_trace_merges_entries_by_node_id(lab):
    junit = lab.write_junit(run("first"), run("second"))
    tree = ET.parse(junit)
    teardown = ET.fromstring(ET.tostring(tree.getroot()[0]))
    teardown.set("name", "teardown of first")
    ET.SubElement(teardown, "error", message="cleanup")
    tree.getroot().append(teardown)
    tree.write(junit)

    result = trace(lab, junit)

    assert result["requirements"][0]["cells"][0]["observed_outcomes"] == ["error"]


@title("Entries without a node id are skipped without hiding later entries")
def test_trace_skips_entries_without_node_id(lab):
    junit = lab.passing_junit()
    tree = ET.parse(junit)
    tree.getroot().insert(0, ET.Element("testcase", name="test_unlabelled"))
    tree.write(junit)

    assert trace(lab, junit)["status"] == "passed"


@title("A protocol scope without requirements is rejected")
def test_trace_rejects_empty_scope(lab):
    with pytest.raises(ValueError, match="Selected protocol has no requirements"):
        trace(lab, phases=["PQ"])


@title("Only requirements in the selected phases are traced")
def test_trace_filters_phases(lab):
    plan = reviewed_plan()
    plan["requirements"].append(plan["requirements"][0] | {"id": "REQ-IQ", "phase": "IQ"})
    lab.write_plan(plan)

    result = trace(lab, lab.passing_junit(), phases=("IQ", "OQ"))

    assert [r["id"] for r in result["requirements"]] == ["REQ-EXAMPLE", "REQ-IQ"]
    assert [r["id"] for r in trace(lab, lab.passing_junit("only.xml"), phases=("IQ", ))["requirements"]] == ["REQ-IQ"]


def build(lab, *junit, attachments=None, output=None):
    output = output or lab.root / "reports/package"
    return output, build_package(
            output, root=lab.root, plan_path=lab.plan_path, junit_files=list(junit or [lab.passing_junit()]),
            phases=["OQ"], attachments=attachments)


@title("A package keeps the exact plan, data, framework, tests, JUnit and attachments, and verifies")
def test_build_package(lab):
    junit = lab.passing_junit()
    attachment = lab.root / "reports/metrics.json"
    attachment.write_text('{"measured": 0.9}')

    output, result = build(lab, junit, attachments=[attachment])

    assert result["status"] == "passed"
    verify_package(output)
    assert {path.relative_to(output).as_posix()
            for path in output.rglob("*") if path.is_file()} == {
            "manifest.json", "summary.md", "traceability.json", "definitions/plan.json", "definitions/data/policy.txt",
            "definitions/src/runtime.py", "definitions/tests/test_example.py", "junit/0-results.xml",
            "attachments/0-metrics.json"}
    assert (output / "definitions/plan.json").read_bytes() == lab.plan_path.read_bytes()
    assert (output / "junit/0-results.xml").read_bytes() == junit.read_bytes()
    assert (output / "attachments/0-metrics.json").read_bytes() == attachment.read_bytes()
    assert json.loads((output / "traceability.json").read_text()) == result
    assert not list(output.parent.glob("qualification-*"))


@title("The summary states scope, result, pending review and no compliance claim")
def test_package_summary(lab):
    output, _ = build(lab, lab.write_junit(run("first")))

    summary = (output / "summary.md").read_text()

    assert summary.startswith(
            "# Educational qualification execution\n\nScope: OQ. Result: incomplete. "
            "Review: pending human review.\n")
    assert "| REQ-EXAMPLE | OQ | high | incomplete |\n" in summary
    assert summary.endswith("does not claim GxP compliance or approval.\n")


@title("The manifest lists each file with its checksum and size, the status and the plan checksum")
def test_package_manifest(lab):
    output, result = build(lab)

    manifest = json.loads((output / "manifest.json").read_text())

    assert (manifest["schema_version"], manifest["educational_only"], manifest["status"],
            manifest["plan_sha256"]) == (1, True, "passed", lab.plan()["sha256"])
    for row in manifest["files"]:
        raw = (output / row["path"]).read_bytes()
        assert (row["sha256"], row["size"]) == (hashlib.sha256(raw).hexdigest(), len(raw))


@title("An existing package is never overwritten")
def test_build_refuses_existing_destination(lab):
    output, _ = build(lab)
    before = (output / "manifest.json").read_bytes()

    with pytest.raises(ValueError, match="Evidence destination already exists"):
        build(lab, output=output)
    assert (output / "manifest.json").read_bytes() == before


@pytest.mark.parametrize(("junit", "attach", "message"), [
        pytest.param(lambda lab: [], False, "Provide distinct JUnit inputs", id="no-junit"),
        pytest.param(lambda lab: [lab.passing_junit()] * 2, False, "Provide distinct JUnit inputs", id="duplicate"),
        pytest.param(
        lambda lab: [lab.passing_junit()], True, "Attachments must be explicitly selected report artifacts",
        id="outside-attachment")])
@title("Invalid package inputs are rejected and leave no partial output [{param_id}]")
def test_build_rejects_invalid_inputs(lab, junit, attach, message):
    output = lab.root / "reports/package"
    files = junit(lab)
    attachments = [lab.plan_path] if attach else None

    with pytest.raises(ValueError, match=message):
        build_package(
                output, root=lab.root, plan_path=lab.plan_path, junit_files=files, phases=["OQ"],
                attachments=attachments)
    assert not output.exists()
    assert not list((lab.root / "reports").glob("qualification-*"))


@pytest.mark.parametrize(("target", "message"), [
        pytest.param("test_data/qualification-plan.json", "Plan changed during packaging", id="plan"),
        pytest.param("reports/results.xml", "JUnit evidence changed during packaging", id="junit"),
        pytest.param("src/runtime.py", "Framework changed during packaging", id="framework"),
        pytest.param("tests/test_example.py", "Test definitions changed during packaging", id="test"),
        pytest.param("test_data/policy.txt", "Data changed during packaging", id="data")])
@title("A change to any input while packaging is refused and leaves no output [{param_id}]")
def test_build_refuses_inputs_changed_during_packaging(lab, monkeypatch, target, message):
    junit = lab.passing_junit()
    original = package.trace_results

    def change_after_tracing(*args, **kwargs):
        result = original(*args, **kwargs)
        path = lab.root / target
        path.write_bytes(path.read_bytes() + b"\nchanged\n")
        return result

    monkeypatch.setattr(package, "trace_results", change_after_tracing)
    output = lab.root / "reports/package"

    with pytest.raises(ValueError, match=message):
        build_package(output, root=lab.root, plan_path=lab.plan_path, junit_files=[junit], phases=["OQ"])
    assert not output.exists()
    assert not list((lab.root / "reports").glob("qualification-*"))


@title("A verified package stays verifiable after moving and deleting the original tree")
def test_package_is_self_contained(lab):
    output, _ = build(lab)
    moved = lab.root.parent / "archived-evidence"
    shutil.move(output, moved)
    shutil.rmtree(lab.root)

    verify_package(moved)


def reseal(output, manifest, name):
    """Update the manifest row of ``name`` to its current bytes."""
    path = output / name
    row = next(r for r in manifest["files"] if r["path"] == name)
    row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=path.stat().st_size)


def edit_json(output, name, **changes):
    path = output / name
    path.write_text(json.dumps(json.loads(path.read_text()) | changes))


@pytest.mark.parametrize(("edit", "message"), [
        pytest.param(lambda out, m: m.update(schema_version=2), "^Invalid evidence manifest$", id="schema"),
        pytest.param(
        lambda out, m: m.update(educational_only=False), "Invalid evidence manifest", id="not-educational"),
        pytest.param(lambda out, m: m.update(files=[]), "Invalid evidence manifest", id="no-files"),
        pytest.param(lambda out, m: m["files"].append(1), "^Invalid evidence file entry$", id="row-not-object"),
        pytest.param(lambda out, m: m["files"][0].update(path=1), "Invalid evidence file entry", id="numeric-path"),
        pytest.param(
        lambda out, m: m["files"][0].update(path="../outside.json"), "^Invalid evidence path$", id="escape"),
        pytest.param(lambda out, m: m["files"][0].update(path="/abs.json"), "Invalid evidence path", id="absolute"),
        pytest.param(lambda out, m: m["files"][0].update(path="./plan.json"), "Invalid evidence path", id="alias"),
        pytest.param(lambda out, m: m["files"][0].update(path="a\\b"), "Invalid evidence path", id="backslash"),
        pytest.param(
        lambda out, m: m["files"].append({"path": "manifest.json"}), "Invalid evidence path", id="manifest-listed"),
        pytest.param(
        lambda out, m: m["files"].append(m["files"][0]), "^Invalid or duplicate evidence path$", id="duplicate"),
        pytest.param(
        lambda out, m: m["files"][0].update(size="1"), "^Evidence checksum or size mismatch: ", id="string-size"),
        pytest.param(
        lambda out, m: m["files"][0].update(sha256="0" * 64), "Evidence checksum or size mismatch", id="checksum"),
        pytest.param(
        lambda out, m: m.update(files=[r for r in m["files"] if r["path"] != "traceability.json"]),
        "missing required definitions or outcomes", id="no-trace"),
        pytest.param(
        lambda out, m:
        (out / "extra.json").write_text("{}"), "^Evidence package contains unlisted files$", id="unlisted-file"),
        pytest.param(
        lambda out, m: m.update(plan_sha256="0" * 64), "^Manifest plan checksum mismatch$", id="plan-checksum"),
        pytest.param(
        lambda out, m: (edit_json(out, "traceability.json", schema_version=True), reseal(out, m, "traceability.json")),
        "^Invalid qualification traceability$", id="trace-schema"),
        pytest.param(
        lambda out, m: (edit_json(out, "traceability.json", input_sha256=[]), reseal(out, m, "traceability.json")),
        "Invalid qualification traceability", id="trace-inputs"),
        pytest.param(
        lambda out, m: m.update(status="passed"), "^Qualification outcomes differ from retained inputs",
        id="claimed-status"),
        pytest.param(
        lambda out, m: (
        m.update(status="passed"), edit_json(out, "traceability.json", status="passed"),
        reseal(out, m, "traceability.json")), "Qualification outcomes differ", id="resealed-trace"),
        pytest.param(
        lambda out, m:
        (edit_json(out, "traceability.json", input_sha256={"x": "0" * 64}), reseal(out, m, "traceability.json")),
        "^Traceability input checksums differ from packaged evidence$", id="input-checksums")])
@title("Verification rejects unsafe, altered or inconsistent packages [{param_id}]")
def test_verify_rejects_altered_package(lab, edit, message):
    output, _ = build(lab, lab.write_junit(run("first")))
    manifest = json.loads((output / "manifest.json").read_text())
    edit(output, manifest)
    (output / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=message):
        verify_package(output)


@title("An edited packaged file fails verification with its path")
def test_verify_rejects_edited_file(lab):
    output, _ = build(lab)
    (output / "summary.md").write_text("changed")

    with pytest.raises(ValueError, match="^Evidence checksum or size mismatch: summary.md$"):
        verify_package(output)


@title("A package without JUnit inputs cannot be verified")
def test_verify_requires_junit(lab):
    output, _ = build(lab)
    manifest = json.loads((output / "manifest.json").read_text())
    for path in (output / "junit").iterdir():
        path.unlink()
    manifest["files"] = [r for r in manifest["files"] if not r["path"].startswith("junit/")]
    (output / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="^Evidence package requires JUnit inputs$"):
        verify_package(output)


def run_cli(monkeypatch, lab, *args):
    monkeypatch.setattr(
            sys, "argv",
            ["qualification", "--root",
            str(lab.root), "--plan",
            str(lab.plan_path), "--phase", "OQ", *map(str, args)])
    return package.main()


@pytest.mark.parametrize(("complete", "code", "status"),
        [pytest.param(True, 0, "passed", id="passed"),
        pytest.param(False, 1, "incomplete", id="incomplete")])
@title("The command builds a verifiable package and exits non-zero unless every requirement passed [{param_id}]")
def test_cli_builds_package(lab, monkeypatch, capsys, complete, code, status):
    junit = lab.passing_junit() if complete else lab.write_junit(run("first"))
    output = lab.root / "reports/package"

    exit_code = run_cli(monkeypatch, lab, "--junit", junit, "--output", output)

    assert exit_code == code
    assert capsys.readouterr().out == f"Educational evidence: {status}; package: {output}\n"
    verify_package(output)


@title("The command packages the given attachments")
def test_cli_attachments(lab, monkeypatch):
    junit = lab.passing_junit()
    attachment = lab.root / "reports/metrics.json"
    attachment.write_text("{}")
    output = lab.root / "reports/package"

    run_cli(monkeypatch, lab, "--junit", junit, "--attach", attachment, "--output", output)

    assert (output / "attachments/0-metrics.json").read_text() == "{}"


@title("By default the command packages the working directory with its reviewed plan")
def test_cli_defaults(lab, monkeypatch):
    junit = lab.passing_junit()
    monkeypatch.chdir(lab.root)
    monkeypatch.setattr(sys, "argv", ["qualification", "--junit", str(junit), "--phase", "OQ", "--output", "package"])

    assert package.main() == 0
    verify_package(lab.root / "package")


@pytest.mark.parametrize("phase", [pytest.param(None, id="no-phase"), pytest.param("DQ", id="unknown-phase")])
@title("The command requires a known protocol phase [{param_id}]")
def test_cli_requires_phase(lab, monkeypatch, capsys, phase):
    phases = ["--phase", phase] if phase else []
    monkeypatch.setattr(sys, "argv", ["qualification", "--junit", "r.xml", "--output", "out", *phases])

    with pytest.raises(SystemExit) as exit:
        package.main()

    assert exit.value.code == 2
    assert "--phase" in capsys.readouterr().err


@pytest.mark.parametrize(
        "arguments",
        [pytest.param(["--output", "out"], id="no-junit"),
        pytest.param(["--junit", "results.xml"], id="no-output")])
@title("The command requires JUnit input and an output [{param_id}]")
def test_cli_requires_arguments(lab, monkeypatch, tmp_path, arguments):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exit:
        run_cli(monkeypatch, lab, *arguments)

    assert exit.value.code == 2


@title("Real pytest execution records requirement, plan, source and node identity in JUnit")
def test_pytest_traceability(framework_pytester, lab):
    child = framework_pytester.path
    (child / "tests").mkdir()
    shutil.copyfile(lab.root / "tests/test_example.py", child / "tests/test_example.py")
    (child / "test_data").mkdir()
    plan = reviewed_plan()
    plan["data_sha256"] = {}
    plan["requirements"][0]["axes"] = {}
    (child / "test_data/qualification-plan.json").write_text(json.dumps(plan))
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.evidence"]')

    result = framework_pytester.runpytest_subprocess("-q", "--junitxml=results.xml")

    result.assert_outcomes(passed=1)
    properties = {p.attrib["name"]: p.attrib["value"] for p in ET.parse(child / "results.xml").findall(".//property")}
    assert properties["test_node_id"] == SELECTOR
    assert (json.loads(properties["requirement_ids"]),
            json.loads(properties["qualification_phases"])) == (["REQ-EXAMPLE"], ["OQ"])
    child_plan = load_plan(child / "test_data/qualification-plan.json", child)
    assert trace_results(child_plan, [child / "results.xml"], ["OQ"])["status"] == "passed"


@title("A plan that references an unknown test stops the pytest run")
def test_pytest_traceability_rejects_unknown_test(framework_pytester, lab):
    child = framework_pytester.path
    (child / "tests").mkdir()
    shutil.copyfile(lab.root / "tests/test_example.py", child / "tests/test_example.py")
    (child / "test_data").mkdir()
    plan = reviewed_plan()
    plan["data_sha256"] = {}
    plan["requirements"][0]["tests"] = ["tests/test_example.py::test_unknown"]
    (child / "test_data/qualification-plan.json").write_text(json.dumps(plan))
    framework_pytester.makeconftest('pytest_plugins = ["llm_testkit.pytest_support.evidence"]')

    result = framework_pytester.runpytest_subprocess("-q")

    assert result.ret == pytest.ExitCode.USAGE_ERROR
