"""Scenario data builders and deterministic test doubles."""

import hashlib
import json
import xml.etree.ElementTree as ET
from copy import deepcopy

from test_support.data.qualification import ROOT as ROOT
from test_support.data.qualification import SELECTOR as SELECTOR


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


def prepare_invalid_plan_case(change, plan, row):
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


def prepare_trace_outcomes_case(rows, status):
    if status != "missing":
        rows.append(("second", status, {}))


def prepare_invalid_package_inputs_case(args, change, junit, path):
    if change == "empty":
        args["junit_files"] = []
    elif change == "duplicate":
        args["junit_files"] = [junit, junit]
    else:
        args["attachments"] = [path]


def prepare_invalid_manifest_case(change, manifest):
    if change == "escape":
        manifest["files"][0]["path"] = "../outside.json"
    elif change == "duplicate":
        manifest["files"].append(manifest["files"][0])
    elif change == "missing":
        manifest["files"] = [r for r in manifest["files"] if r["path"] != "traceability.json"]
    else:
        manifest["schema_version"] = 2


def make_change_after_parsing_stub(change, junit, original_trace, path, root):
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

    return change_after_parsing


def prepare_conflicting_inline_provenance_cannot_pass_case(duplicate, props, stale_first):
    if stale_first:
        props.insert(0, duplicate)
    else:
        props.append(duplicate)


def prepare_package_verification_binds_outcomes_to_inputs_case(change, manifest, output):
    if change == "unlisted":
        (output / "extra.json").write_text("{}")
    elif change == "trace-schema":
        trace_path = output / "traceability.json"
        trace = json.loads(trace_path.read_text())
        trace["schema_version"] = True
        trace_path.write_text(json.dumps(trace))
        row = next(r for r in manifest["files"] if r["path"] == "traceability.json")
        row.update(
            sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            size=trace_path.stat().st_size,
        )
    elif change == "plan":
        manifest["plan_sha256"] = "0" * 64
    else:
        manifest["status"] = "passed"
        if change == "resealed-trace":
            trace_path = output / "traceability.json"
            trace = json.loads(trace_path.read_text())
            trace["status"] = "passed"
            trace_path.write_text(json.dumps(trace))
            row = next(r for r in manifest["files"] if r["path"] == "traceability.json")
            row.update(
                sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),
                size=trace_path.stat().st_size,
            )


def prepare_plan_rejects_ambiguous_paths_and_phase_types_case(change, data):
    if change == "selector-alias":
        data["requirements"][0]["tests"] = ["tests/../tests/test_example.py::test_example"]
    elif change == "data-alias":
        checksum = data["data_sha256"].pop("policy.txt")
        data["data_sha256"]["./policy.txt"] = checksum
    else:
        data["requirements"][0]["phase"] = []


def make_teardown_entries(change):
    duplicate = (
        ("first", "error", {})
        if change == "error"
        else (
            "first",
            "passed",
            {"requirement_ids": "[]"},
        )
    )

    return duplicate
