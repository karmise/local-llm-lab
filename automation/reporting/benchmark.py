"""Summarize the planned matrix, preserving missing results and judge failures."""

import math
from collections import Counter
from statistics import mean

from helpers.scores import require_score
from models.benchmark import configuration_hash
from reporting.gates import METRICS


def row_status(row: dict) -> str:
    statuses = {d["status"] for d in row.get("dimensions", [])}
    if row.get("error") or row.get("generation_status") in ("error", "skipped"):
        return "error"
    if "error" in statuses or not statuses:
        return "error"
    if "failed" in statuses or row.get("generation_status") == "failed":
        return "failed"
    return "passed"


def _group(expected: list[dict], rows: dict[tuple[str, str], dict]) -> dict:
    selected = [rows.get((r["case_id"], r["model"])) for r in expected]
    outcomes = Counter(row_status(row) if row is not None else "missing" for row in selected)
    summary = {
            "planned": len(expected),
            "passed": outcomes["passed"],
            "failed": outcomes["failed"],
            "errors": outcomes["error"],
            "missing": outcomes["missing"],
            "pass_rate": outcomes["passed"] / len(expected),
            "metrics": {}}
    durations = [
            row["answer_request_seconds"] for row in selected if row is not None and "answer_request_seconds" in row]
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError("Benchmark answer durations must be finite positive numbers")
    summary["answer_timing"] = {
            "measured": len(durations),
            "unavailable": len(expected) - len(durations),
            "mean_seconds": mean(durations) if durations else None,
            "minimum_seconds": min(durations) if durations else None,
            "maximum_seconds": max(durations) if durations else None,
            "scope": (
            "Completed chat API request, including retrieval/model loading/generation; excludes "
            "fixture setup, judge calls and cleanup")}
    for metric in sorted(METRICS):
        eligible = [r for r in expected if r["category"] != "missing_information"]
        dimensions = [
                d for planned in eligible
                for d in rows.get((planned["case_id"], planned["model"]), {}).get("dimensions", [])
                if d.get("metric") == metric and d["status"] in ("passed", "failed", "measured")]
        values = [d["value"] for d in dimensions]
        summary["metrics"][metric] = {
                "eligible": len(eligible),
                "measured": len(values),
                "unavailable": len(eligible) - len(values),
                "not_applicable": len(expected) - len(eligible),
                "below_minimum": sum(d["status"] == "failed" for d in dimensions),
                "mean": mean(values) if values else None,
                "minimum": min(values) if values else None}
    return summary


def _check_dimensions(row: dict, minima: dict) -> None:
    """A completed row has two reviewed checks and four metric outcomes that agree with their scores."""
    dimensions = row["dimensions"]
    metrics = [d.get("metric") for d in dimensions if d.get("metric")]
    if len(metrics) != 4 or set(metrics) != METRICS or len(dimensions) != 6:
        raise ValueError("Benchmark result requires two checks and four metric outcomes")
    for dimension in dimensions:
        if dimension["status"] not in ("passed", "failed", "measured", "error", "not_applicable"):
            raise ValueError("Unknown benchmark dimension status")
        metric = dimension.get("metric")
        if not metric:
            if dimension["status"] == "not_applicable":
                raise ValueError("Reviewed answer/source checks are always required")
            continue
        refusal = row["category"] == "missing_information"
        if refusal != (dimension["status"] == "not_applicable"):
            raise ValueError("Only refusal metrics may be not applicable")
        if dimension["status"] in ("passed", "failed", "measured"):
            require_score(dimension["value"], f"Benchmark {metric} score")
            if dimension.get("minimum") != minima[metric]:
                raise ValueError("Benchmark threshold differs from the manifest")
            outcome = (
                    "measured"
                    if minima[metric] is None else "passed" if dimension["value"] >= minima[metric] else "failed")
            if dimension["status"] != outcome:
                raise ValueError("Benchmark status differs from its metric score")


def _check_generation_setup(row: dict, configurations: dict[str, str], digests: dict[str, str]) -> None:
    """Every row of a model used the same weights, and all models used the same workspace settings."""
    model = row["model"]
    configuration = configuration_hash(row["workspace_configuration"])
    digest = row["model_digest"]
    if model in configurations and configurations[model] != configuration:
        raise ValueError("Workspace configuration changed within one model's benchmark")
    if configurations and configuration not in configurations.values():
        raise ValueError("Generation models were benchmarked with different workspace settings")
    if model in digests and digests[model] != digest:
        raise ValueError("Generation model digest changed within the benchmark")
    configurations[model], digests[model] = configuration, digest


def _judge_controls_matched(calibration: dict, manifest: dict) -> bool:
    """The judge passed every declared control, with the judge and control set the manifest names."""
    return (
            calibration.get("status") == "matched" and calibration.get("judge_model") == manifest["judge_model"]
            and calibration.get("controls_sha256") == manifest["controls_sha256"]
            and calibration.get("control_ids") == manifest["control_ids"]
            and bool(calibration.get("judge_model_digest"))
            and len(calibration.get("results", [])) == len(manifest["control_ids"])
            and all(r.get("status") == "matched" for r in calibration["results"]))


def summarize(manifest: dict, results: list[dict], calibration: dict) -> dict:
    expected = manifest["expected_rows"]
    identities = {(r["case_id"], r["model"]): r for r in expected}
    if not expected or len(identities) != len(expected):
        raise ValueError("Benchmark manifest requires a unique, nonempty matrix")
    rows = {}
    configurations = {}
    digests = {}
    minima = manifest["quality_gates"]["minimum_scores"]
    if set(minima) != METRICS:
        raise ValueError("Benchmark requires all four metric thresholds")
    for row in results:
        identity = (row["case_id"], row["model"])
        if identity not in identities or identity in rows:
            raise ValueError("Unknown or duplicate benchmark result")
        if row["category"] != identities[identity]["category"]:
            raise ValueError("Benchmark result category mismatch")
        rows[identity] = row
        if row.get("error"):
            continue
        _check_dimensions(row, minima)
        _check_generation_setup(row, configurations, digests)
    summary = _group(expected, rows)
    calibration_ok = _judge_controls_matched(calibration, manifest)
    status = (
            "error" if summary["errors"] or summary["missing"] or not calibration_ok else
            "failed" if summary["failed"] else "checks_passed")
    return {
            "schema_version": 1,
            "status": status,
            "manifest": manifest,
            "interpretation": (
            "Experimental thresholds and a curated sample; human review pending. A mean cannot hide "
            "failed, missing or invalid cases. No population accuracy or clinical validity claim."),
            "summary": summary,
            "models": {
            model: _group([r for r in expected if r["model"] == model], rows)
            for model in dict.fromkeys(r["model"] for r in expected)},
            "categories": {
            category: _group([r for r in expected if r["category"] == category], rows)
            for category in dict.fromkeys(r["category"] for r in expected)},
            "calibration": calibration,
            "judge_controls_matched": calibration_ok,
            "results": results,
            "missing_rows": [r for key, r in identities.items() if key not in rows]}
