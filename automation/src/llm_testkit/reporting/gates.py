"""Fail closed on missing, invalid or below-threshold quality evidence."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from llm_testkit import assertions

METRICS = frozenset({"faithfulness", "factual_correctness", "context_precision", "context_recall"})


def load_quality_gates(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    data = json.loads(raw)
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("Unsupported quality gates schema")
    if data.get("calibration") != "experimental":
        raise ValueError("This lab supports explicitly experimental gates only")
    if any(not isinstance(data.get(k), str) or not data[k].strip() for k in ("version", "rationale")):
        raise ValueError("Quality gates require version and rationale")
    minima = data.get("minimum_scores")
    if not isinstance(minima, dict) or set(minima) != METRICS:
        raise ValueError("Quality gates must define all four semantic metrics")
    for value in minima.values():
        assertions.assert_quality_score(value)
    return {**data, "sha256": hashlib.sha256(raw).hexdigest()}


def apply_quality_gates(report: dict[str, Any], path: Path) -> dict[str, Any]:
    gated = deepcopy(report)
    gates = load_quality_gates(path)
    gated["quality_gates"] = gates
    gated["interpretation"] = gates["rationale"]
    seen = set()
    for dimension in gated["dimensions"]:
        metric = dimension.get("metric")
        if metric not in METRICS:
            continue
        if metric in seen:
            raise ValueError("Duplicate quality metric dimension")
        seen.add(metric)
        minimum = gates["minimum_scores"][metric]
        dimension["name"] = f"{metric} gate (minimum {minimum})"
        # Failed/error dimensions stay failed/error; no numeric value can hide invalid evidence.
        if dimension["status"] == "measured":
            details = dimension["details"]
            assertions.assert_quality_score(details["value"])
            details["threshold"] = minimum
            dimension["status"] = "passed" if details["value"] >= minimum else "failed"
            if dimension["status"] == "failed":
                dimension["error"] = f"Score {details['value']} is below minimum {minimum}"
        elif dimension["status"] not in ("failed", "error"):
            dimension.update(status="error", error="Gate requires validated measured evidence")
    for missing in sorted(METRICS - seen):
        gated["dimensions"].append({
                "name": f"{missing} gate",
                "metric": missing,
                "status": "error",
                "error": "Required metric evidence was not supplied"})
    statuses = {d["status"] for d in gated["dimensions"]}
    gated["status"] = ("error" if "error" in statuses else "failed" if "failed" in statuses else "checks_passed")
    return gated
