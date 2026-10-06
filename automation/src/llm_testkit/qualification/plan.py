"""Reviewed requirement-to-test mapping, validated against real test definitions."""

import ast
import hashlib
import json
import re
from itertools import product
from pathlib import Path
from typing import Any

PHASES = frozenset({"IQ", "OQ", "PQ"})


def framework_sources(root: Path) -> list[Path]:
    files = [f for f in (root / "src").rglob("*") if f.suffix in (".py", ".cjs")]
    files += list(root.glob("requirements*.lock"))
    files += [root / "pyproject.toml", root / "tests/conftest.py"]
    return sorted({f for f in files if f.is_file()})


def framework_checksum(root: Path) -> str:
    source = hashlib.sha256()
    for file in framework_sources(root):
        source.update(str(file.relative_to(root)).encode())
        source.update(b"\0")
        source.update(file.read_bytes())
        source.update(b"\0")
    return source.hexdigest()


def load_plan(path: Path, root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    plan = json.loads(raw)
    if (
        not isinstance(plan, dict)
        or type(plan.get("schema_version")) is not int
        or plan["schema_version"] != 1
        or plan.get("educational_only") is not True
    ):
        raise ValueError("Plan must use the supported educational schema")
    if not isinstance(plan.get("version"), str) or not plan["version"].strip():
        raise ValueError("Plan requires a version")
    data = plan.get("data_sha256", {})
    if not isinstance(data, dict) or any(
        not isinstance(name, str)
        or not isinstance(checksum, str)
        or not re.fullmatch(r"[a-f0-9]{64}", checksum)
        for name, checksum in data.items()
    ):
        raise ValueError("Data baselines must use file names and SHA-256 checksums")
    for name, checksum in data.items():
        target = (root / "test_data" / name).resolve()
        if (
            not target.is_relative_to((root / "test_data").resolve())
            or hashlib.sha256(target.read_bytes()).hexdigest() != checksum
        ):
            raise ValueError("Plan data baseline changed; review expectations and version: " + name)
    rows = plan.get("requirements")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Plan requires requirements")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Requirements must be objects")
        identifier = row.get("id")
        if (
            not isinstance(identifier, str)
            or not re.fullmatch(r"REQ-[A-Z0-9-]+", identifier)
            or identifier in seen
        ):
            raise ValueError("Invalid or duplicate requirement id")
        seen.add(identifier)
        if row.get("phase") not in PHASES or row.get("risk") not in ("low", "medium", "high"):
            raise ValueError("Unknown qualification phase or risk")
        for field in ("description", "acceptance", "rationale"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError("Requirement criteria must be explicit")
        selectors = row.get("tests")
        if (
            not isinstance(selectors, list)
            or not selectors
            or any(not isinstance(s, str) for s in selectors)
            or len(set(selectors)) != len(selectors)
        ):
            raise ValueError("Requirement must map to distinct test selectors")
        for selector in selectors:
            parts = selector.split("::")
            if len(parts) != 2 or not parts[0].startswith("tests/") or not parts[0].endswith(".py"):
                raise ValueError("Use unparameterized test function selectors")
            target = (root / parts[0]).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError("Test selector escapes automation root")
            functions = {
                node.name
                for node in ast.parse(target.read_text()).body
                if isinstance(node, ast.FunctionDef)
            }
            if parts[1] not in functions:
                raise ValueError("Requirement references an unknown test: " + selector)
        axes = row.get("axes", {})
        if not isinstance(axes, dict) or any(
            not isinstance(k, str)
            or not isinstance(v, list)
            or not v
            or any(not isinstance(x, str) or not x for x in v)
            or len(set(v)) != len(v)
            for k, v in axes.items()
        ):
            raise ValueError("Requirement axes must declare nonempty distinct string values")
    return {
        **plan,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "framework_source_sha256": framework_checksum(root),
        "test_source_sha256": {
            selector: hashlib.sha256((root / selector.split("::")[0]).read_bytes()).hexdigest()
            for row in rows
            for selector in row["tests"]
        },
    }


def matching_requirements(
    plan: dict[str, Any], node_id: str, properties: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    selector = node_id.split("[", 1)[0]
    return [
        row
        for row in plan["requirements"]
        if selector in row["tests"]
        and (
            properties is None
            or all(properties.get(k) in values for k, values in row.get("axes", {}).items())
        )
    ]


def expected_cells(requirement: dict[str, Any]) -> list[tuple[str, tuple[tuple[str, str], ...]]]:
    axes = requirement.get("axes", {})
    names = sorted(axes)
    return [
        (selector, tuple(zip(names, values, strict=True)))
        for selector in requirement["tests"]
        for values in product(*(axes[n] for n in names))
    ]
