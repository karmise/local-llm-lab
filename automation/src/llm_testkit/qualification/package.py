"""Build a scoped educational evidence package, retaining failures and missing coverage."""

import argparse
import hashlib
import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.qualification.plan import (
    PHASES,
    expected_cells,
    framework_checksum,
    framework_sources,
    load_plan,
)


def trace_results(
    plan: dict[str, Any], junit_files: list[Path], phases: list[str]
) -> dict[str, Any]:
    if not phases or len(set(phases)) != len(phases) or not set(phases) <= PHASES:
        raise ValueError("Select distinct IQ/OQ/PQ protocol phases")
    input_sha256 = {}
    observations = {}
    deviations = []
    for path in junit_files:
        # Each input is a run. Duplicate testcase elements within a run may represent teardown errors.
        raw = path.read_bytes()
        input_sha256[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        for row in ET.fromstring(raw).iter("testcase"):
            props = {
                p.attrib["name"]: p.attrib.get("value", "")
                for p in row.findall("./properties/property")
            }
            node = props.get("test_node_id")
            if not node:
                continue
            identity = (str(path.resolve()), node)
            entry = observations.setdefault(
                identity, {"node": node, "properties": {}, "status": "passed", "details": []}
            )
            for name, value in props.items():
                if name in entry["properties"] and entry["properties"][name] != value:
                    entry["status"] = "error"
                    entry["details"].append("Conflicting metadata: " + name)
                entry["properties"][name] = value
            priorities = {"passed": 0, "failed": 1, "skipped": 2, "error": 3}
            for tag, status in [("failure", "failed"), ("skipped", "skipped"), ("error", "error")]:
                child = row.find(tag)
                if child is not None:
                    if priorities[status] > priorities[entry["status"]]:
                        entry["status"] = status
                    entry["details"].append(
                        {
                            "kind": status,
                            "message": child.attrib.get("message", ""),
                            "text": child.text or "",
                        }
                    )
    requirements = []
    for req in plan["requirements"]:
        if req["phase"] not in phases:
            continue
        cells = []
        for selector, axis in expected_cells(req):
            matching = [
                e
                for e in observations.values()
                if e["node"].split("[", 1)[0] == selector
                and all(e["properties"].get(k) == v for k, v in axis)
            ]
            outcomes = []
            for entry in matching:
                p = entry["properties"]
                try:
                    ids = json.loads(p.get("requirement_ids", "[]"))
                    bound = (
                        p.get("qualification_plan_sha256") == plan["sha256"]
                        and isinstance(ids, list)
                        and req["id"] in ids
                        and p.get("framework_source_sha256") == plan["framework_source_sha256"]
                        and p.get("test_source_sha256") == plan["test_source_sha256"][selector]
                    )
                except (ValueError, TypeError):
                    bound = False
                outcome = entry["status"] if bound else "unbound"
                outcomes.append(outcome)
                if outcome != "passed":
                    deviations.append(
                        {
                            "requirement": req["id"],
                            "node": entry["node"],
                            "status": outcome,
                            "details": entry["details"],
                        }
                    )
            status = (
                "missing"
                if not outcomes
                else "failed"
                if "failed" in outcomes
                else "incomplete"
                if any(o != "passed" for o in outcomes)
                else "passed"
            )
            # A pass from another run never erases an earlier failure/error/skip.
            cells.append(
                {
                    "test": selector,
                    "axes": dict(axis),
                    "status": status,
                    "observed_outcomes": outcomes,
                }
            )
        statuses = {c["status"] for c in cells}
        status = (
            "failed"
            if "failed" in statuses
            else "incomplete"
            if statuses != {"passed"}
            else "passed"
        )
        requirements.append(
            {
                "id": req["id"],
                "phase": req["phase"],
                "risk": req["risk"],
                "acceptance": req["acceptance"],
                "status": status,
                "cells": cells,
            }
        )
    statuses = {r["status"] for r in requirements}
    if not requirements:
        raise ValueError("Selected protocol has no requirements")
    status = (
        "failed" if "failed" in statuses else "incomplete" if statuses != {"passed"} else "passed"
    )
    return {
        "schema_version": 1,
        "educational_only": True,
        "input_sha256": input_sha256,
        "protocol_scope": phases,
        "status": status,
        "plan_version": plan["version"],
        "plan_sha256": plan["sha256"],
        "requirements": requirements,
        "deviations": deviations,
        "review_status": "pending human review",
        "compliance_claim": False,
        "interpretation": "Execution and traceability within declared educational protocol scope; no electronic signature, formal approval or GxP certification",
    }


def build_package(
    destination: Path,
    *,
    root: Path,
    plan_path: Path,
    junit_files: list[Path],
    phases: list[str],
    attachments: list[Path] | None = None,
) -> dict[str, Any]:
    if destination.exists():
        raise ValueError("Evidence destination already exists")
    if not junit_files or len({p.resolve() for p in junit_files}) != len(junit_files):
        raise ValueError("Provide distinct JUnit inputs")
    plan = load_plan(plan_path, root)
    trace = trace_results(plan, junit_files, phases)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="qualification-", dir=destination.parent))
    try:
        files = []
        # Copy exact definitions and input evidence; checksum the copied bytes.
        sources = [("definitions/plan.json", plan_path)]
        sources += [
            ("definitions/data/" + name, root / "test_data" / name)
            for name in plan.get("data_sha256", {})
        ]
        sources += [("definitions/" + str(p.relative_to(root)), p) for p in framework_sources(root)]
        sources += [
            ("definitions/" + selector.split("::")[0], root / selector.split("::")[0])
            for selector in sorted({s for r in plan["requirements"] for s in r["tests"]})
        ]
        sources = list(dict.fromkeys(sources))
        sources += [(f"junit/{i}-{p.name}", p) for i, p in enumerate(junit_files)]
        sources += [(f"attachments/{i}-{p.name}", p) for i, p in enumerate(attachments or [])]
        for relative, path in sources:
            if relative.startswith("attachments/") and not path.resolve().is_relative_to(
                (root / "reports").resolve()
            ):
                raise ValueError("Attachments must be explicitly selected report artifacts")
            raw = path.read_bytes()
            checksum = hashlib.sha256(raw).hexdigest()
            if relative == "definitions/plan.json" and checksum != plan["sha256"]:
                raise ValueError("Plan changed during packaging")
            if (
                relative.startswith("junit/")
                and checksum != trace["input_sha256"][str(path.resolve())]
            ):
                raise ValueError("JUnit evidence changed during packaging")
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            files.append(
                {"path": relative, "sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
            )
        definitions = temporary / "definitions"
        if framework_checksum(definitions) != plan["framework_source_sha256"]:
            raise ValueError("Framework changed during packaging")
        for selector, checksum in plan["test_source_sha256"].items():
            if (
                hashlib.sha256((definitions / selector.split("::")[0]).read_bytes()).hexdigest()
                != checksum
            ):
                raise ValueError("Test definitions changed during packaging")
        for name, checksum in plan.get("data_sha256", {}).items():
            if hashlib.sha256((definitions / "data" / name).read_bytes()).hexdigest() != checksum:
                raise ValueError("Data changed during packaging")
        write_sample(temporary / "traceability.json", trace)
        files.append(
            {
                "path": "traceability.json",
                "sha256": hashlib.sha256(
                    (temporary / "traceability.json").read_bytes()
                ).hexdigest(),
                "size": (temporary / "traceability.json").stat().st_size,
            }
        )
        summary = "# Educational qualification execution\n\n"
        summary += f"Scope: {', '.join(phases)}. Result: {trace['status']}. Review: pending human review.\n\n"
        summary += "| Requirement | Phase | Risk | Result |\n| --- | --- | --- | --- |\n"
        summary += "".join(
            f"| {r['id']} | {r['phase']} | {r['risk']} | {r['status']} |\n"
            for r in trace["requirements"]
        )
        summary += "\nSee traceability.json for missing cells and deviations. This package does not claim GxP compliance or approval.\n"
        (temporary / "summary.md").write_text(summary)
        files.append(
            {
                "path": "summary.md",
                "sha256": hashlib.sha256(summary.encode()).hexdigest(),
                "size": len(summary.encode()),
            }
        )
        write_sample(
            temporary / "manifest.json",
            {
                "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "educational_only": True,
                "files": files,
                "status": trace["status"],
                "plan_sha256": plan["sha256"],
            },
        )
        verify_package(temporary)
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return trace


def verify_package(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("educational_only") is not True
        or not manifest.get("files")
    ):
        raise ValueError("Invalid evidence manifest")
    seen = set()
    for row in manifest["files"]:
        relative = row["path"]
        path = (directory / relative).resolve()
        if relative in seen or not path.is_relative_to(directory.resolve()):
            raise ValueError("Invalid or duplicate evidence path")
        seen.add(relative)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"] or len(raw) != row["size"]:
            raise ValueError("Evidence checksum or size mismatch: " + relative)
    if not {"traceability.json", "summary.md", "definitions/plan.json"} <= seen:
        raise ValueError("Evidence package is missing required definitions or outcomes")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path, action="append", required=True)
    parser.add_argument("--phase", choices=sorted(PHASES), action="append", required=True)
    parser.add_argument("--plan", type=Path, default=Path("test_data/qualification-plan.json"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--attach", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    trace = build_package(
        args.output,
        root=args.root,
        plan_path=args.plan,
        junit_files=args.junit,
        phases=args.phase,
        attachments=args.attach,
    )
    print(f"Educational evidence: {trace['status']}; package: {args.output}")
    return 0 if trace["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
