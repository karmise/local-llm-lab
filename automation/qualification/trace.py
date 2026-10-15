"""Trace JUnit results to the reviewed requirements, keeping failures, unbound runs and missing cells."""

import json
from pathlib import Path
from typing import Any

from qualification.plan import PHASES, expected_cells
from reporting.junit import read_junit


def combined_status(statuses: set[str]) -> str:
    """Failed if anything failed, passed only if everything passed, otherwise incomplete."""
    return "failed" if "failed" in statuses else "passed" if statuses == {"passed"} else "incomplete"


def _observations(junit_files: list[Path]) -> tuple[dict[str, str], dict[tuple[str, str], dict[str, Any]]]:
    """Input checksums, and every JUnit entry that names its test node, keyed by run and node."""
    input_sha256 = {}
    observations = {}
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
                    "details": [*details, *entry.details]}
    return input_sha256, observations


def _is_bound(properties: dict[str, str], plan: dict[str, Any], requirement: str, selector: str) -> bool:
    """The run recorded this plan, requirement, framework and test source, so its result counts."""
    try:
        ids = json.loads(properties.get("requirement_ids", "[]"))
        return (
                properties.get("qualification_plan_sha256") == plan["sha256"] and isinstance(ids, list)
                and requirement in ids and properties.get("framework_source_sha256") == plan["framework_source_sha256"]
                and properties.get("test_source_sha256") == plan["test_source_sha256"][selector])
    except (ValueError, TypeError):
        return False


def trace_results(plan: dict[str, Any], junit_files: list[Path], phases: list[str]) -> dict[str, Any]:
    if not phases or len(set(phases)) != len(phases) or not set(phases) <= PHASES:
        raise ValueError("Select distinct IQ/OQ/PQ protocol phases")
    input_sha256, observations = _observations(junit_files)
    deviations = []
    requirements = []
    for req in plan["requirements"]:
        if req["phase"] not in phases:
            continue
        cells = []
        for selector, axis in expected_cells(req):
            matching = [
                    e for e in observations.values()
                    if e["node"].split("[", 1)[0] == selector and all(e["properties"].get(k) == v for k, v in axis)]
            outcomes = []
            for entry in matching:
                outcome = entry["status"] if _is_bound(entry["properties"], plan, req["id"], selector) else "unbound"
                outcomes.append(outcome)
                if outcome != "passed":
                    deviations.append({
                            "requirement": req["id"],
                            "node": entry["node"],
                            "status": outcome,
                            "details": entry["details"]})
            # A pass from another run never erases an earlier failure/error/skip.
            status = combined_status(set(outcomes)) if outcomes else "missing"
            cells.append({"test": selector, "axes": dict(axis), "status": status, "observed_outcomes": outcomes})
        requirements.append({
                "id": req["id"],
                "phase": req["phase"],
                "risk": req["risk"],
                "acceptance": req["acceptance"],
                "status": combined_status({c["status"]
                for c in cells}),
                "cells": cells})
    if not requirements:
        raise ValueError("Selected protocol has no requirements")
    return {
            "schema_version": 1,
            "educational_only": True,
            "input_sha256": input_sha256,
            "protocol_scope": phases,
            "status": combined_status({r["status"]
            for r in requirements}),
            "plan_version": plan["version"],
            "plan_sha256": plan["sha256"],
            "requirements": requirements,
            "deviations": deviations,
            "review_status": "pending human review",
            "compliance_claim": False,
            "interpretation": (
            "Execution and traceability within declared educational protocol scope; no electronic "
            "signature, formal approval or GxP certification")}
