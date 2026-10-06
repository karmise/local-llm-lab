"""Build a scoped educational evidence package, retaining failures and missing coverage."""

import argparse
import hashlib
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.qualification.plan import (
    PHASES,
    expected_cells,
    framework_checksum,
    framework_sources,
    load_plan,
)
from llm_testkit.reporting.junit import read_junit


def trace_results(
    plan: dict[str, Any], junit_files: list[Path], phases: list[str]
) -> dict[str, Any]:
    if not phases or len(set(phases)) != len(phases) or not set(phases) <= PHASES:
        raise ValueError("Select distinct IQ/OQ/PQ protocol phases")
    input_sha256 = {}
    observations = {}
    deviations = []
    for path in junit_files:
        # Each file is an independent run; phase entries within it share node identity.
        junit = read_junit(path, identity_property="test_node_id")
        input_sha256[str(path.resolve())] = junit.sha256
        for entry in junit.cases.values():
            node = entry.properties.get("test_node_id")
            if not node:
                continue
            details = ["Conflicting metadata: " + name for name in sorted(entry.conflicts)]
            observations[(str(path.resolve()), node)] = {
                "node": node,
                "properties": entry.properties,
                "status": entry.status,
                "details": [*details, *entry.details],
            }
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
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("educational_only") is not True
        or not isinstance(manifest.get("files"), list)
        or not manifest.get("files")
    ):
        raise ValueError("Invalid evidence manifest")
    seen = set()
    for row in manifest["files"]:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ValueError("Invalid evidence file entry")
        relative = row["path"]
        parts = PurePosixPath(relative)
        if (
            parts.is_absolute()
            or ".." in parts.parts
            or parts.as_posix() != relative
            or "\\" in relative
            or relative == "manifest.json"
        ):
            raise ValueError("Invalid evidence path")
        path = (directory / relative).resolve()
        if relative in seen or not path.is_relative_to(directory.resolve()):
            raise ValueError("Invalid or duplicate evidence path")
        seen.add(relative)
        raw = path.read_bytes()
        if (
            hashlib.sha256(raw).hexdigest() != row.get("sha256")
            or type(row.get("size")) is not int
            or len(raw) != row["size"]
        ):
            raise ValueError("Evidence checksum or size mismatch: " + relative)
    if not {"traceability.json", "summary.md", "definitions/plan.json"} <= seen:
        raise ValueError("Evidence package is missing required definitions or outcomes")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != seen | {"manifest.json"}:
        raise ValueError("Evidence package contains unlisted files")
    definitions = directory / "definitions"
    plan = load_plan(definitions / "plan.json", definitions, data_root=definitions / "data")
    if manifest.get("plan_sha256") != plan["sha256"]:
        raise ValueError("Manifest plan checksum mismatch")
    trace = json.loads((directory / "traceability.json").read_text())
    if (
        not isinstance(trace, dict)
        or type(trace.get("schema_version")) is not int
        or trace["schema_version"] != 1
        or trace.get("educational_only") is not True
        or not isinstance(trace.get("input_sha256"), dict)
    ):
        raise ValueError("Invalid qualification traceability")
    inputs = [
        directory / row["path"] for row in manifest["files"] if row["path"].startswith("junit/")
    ]
    if not inputs:
        raise ValueError("Evidence package requires JUnit inputs")
    expected = trace_results(plan, inputs, trace.get("protocol_scope", []))
    # Original absolute paths change when a package moves; compare the retained bytes in run order.
    if list(trace["input_sha256"].values()) != list(expected["input_sha256"].values()):
        raise ValueError("Traceability input checksums differ from packaged evidence")
    trace_payload = {k: v for k, v in trace.items() if k != "input_sha256"}
    expected_payload = {k: v for k, v in expected.items() if k != "input_sha256"}
    if trace_payload != expected_payload or manifest.get("status") != expected["status"]:
        raise ValueError("Qualification outcomes differ from retained inputs and definitions")


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
