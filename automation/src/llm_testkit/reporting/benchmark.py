"""Summarize the planned matrix, preserving missing results and judge failures."""

import json
import math
from collections import Counter
from statistics import mean

from llm_testkit import assertions
from llm_testkit.core.scores import require_score
from llm_testkit.datasets.benchmark import configuration_hash
from llm_testkit.reporting.gates import METRICS
from llm_testkit.reporting.steps import attach_text, set_metadata, step


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
            "measured":
            len(durations),
            "unavailable":
            len(expected) - len(durations),
            "mean_seconds":
            mean(durations) if durations else None,
            "minimum_seconds":
            min(durations) if durations else None,
            "maximum_seconds":
            max(durations) if durations else None,
            "scope":
            "Completed chat API request, including retrieval/model loading/generation; excludes fixture setup, judge calls and cleanup"
    }
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
        dimensions = row["dimensions"]
        metrics = [d.get("metric") for d in dimensions if d.get("metric")]
        if len(metrics) != 4 or set(metrics) != METRICS or len(dimensions) != 6:
            raise ValueError("Benchmark result requires two checks and four metric outcomes")
        for dimension in dimensions:
            if dimension["status"] not in ("passed", "failed", "measured", "error", "not_applicable"):
                raise ValueError("Unknown benchmark dimension status")
            metric = dimension.get("metric")
            if metric:
                refusal = row["category"] == "missing_information"
                if refusal != (dimension["status"] == "not_applicable"):
                    raise ValueError("Only refusal metrics may be not applicable")
                if dimension["status"] in ("passed", "failed", "measured"):
                    require_score(dimension["value"], f"Benchmark {metric} score")
                    if dimension.get("minimum") != minima[metric]:
                        raise ValueError("Benchmark threshold differs from the manifest")
                    outcome = (
                            "measured" if minima[metric] is None else
                            "passed" if dimension["value"] >= minima[metric] else "failed")
                    if dimension["status"] != outcome:
                        raise ValueError("Benchmark status differs from its metric score")
            elif dimension["status"] == "not_applicable":
                raise ValueError("Reviewed answer/source checks are always required")
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
    summary = _group(expected, rows)
    calibration_ok = (
            calibration.get("status") == "matched" and calibration.get("judge_model") == manifest["judge_model"]
            and calibration.get("controls_sha256") == manifest["controls_sha256"]
            and calibration.get("control_ids") == manifest["control_ids"]
            and bool(calibration.get("judge_model_digest"))
            and len(calibration.get("results", [])) == len(manifest["control_ids"])
            and all(r.get("status") == "matched" for r in calibration["results"]))
    status = (
            "error" if summary["errors"] or summary["missing"] or not calibration_ok else
            "failed" if summary["failed"] else "checks_passed")
    return {
            "schema_version": 1,
            "status": status,
            "manifest": manifest,
            "interpretation":
            "Experimental thresholds and a curated sample; human review pending. A mean cannot hide failed, missing or invalid cases. No population accuracy or clinical validity claim.",
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


def markdown(report: dict) -> str:
    lines = [
            "# Policy quality benchmark", "", f"Status: **{report['status']}**", "", report["interpretation"], "",
            "## Results by generation model", "",
            "| Model | Planned | Passed | Failed | Errors | Missing | Pass rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, summary in report["models"].items():
        lines.append(
                f"| {model} | {summary['planned']} | {summary['passed']} | {summary['failed']} | {summary['errors']} | {summary['missing']} | {summary['pass_rate']:.1%} |"
        )
    lines.extend([
            "", "## Answer request timing", "",
            "Descriptive timings include retrieval, model loading and generation. They exclude fixture setup, judge calls and cleanup. One observation per case is not a latency SLO or a statistically reliable speed ranking.",
            "", "| Model | Measured / planned | Unavailable | Mean seconds | Min seconds | Max seconds |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for model, summary in report["models"].items():
        timing = summary["answer_timing"]
        values = [
                "N/A" if timing[key] is None else f"{timing[key]:.3f}"
                for key in ("mean_seconds", "minimum_seconds", "maximum_seconds")]
        lines.append(
                f"| {model} | {timing['measured']} / {summary['planned']} | {timing['unavailable']} | "
                f"{' | '.join(values)} |")
    lines.extend([
            "", "## Semantic metrics by model", "",
            "Means cover available measurements only. Check unavailable counts before comparing.", "",
            "| Model | Metric | Mean | Minimum | Measured / eligible | Unavailable | Below threshold | N/A |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for model, summary in report["models"].items():
        for metric, values in summary["metrics"].items():
            avg = "N/A" if values["mean"] is None else f"{values['mean']:.3f}"
            minimum = "N/A" if values["minimum"] is None else f"{values['minimum']:.3f}"
            lines.append(
                    f"| {model} | {metric} | {avg} | {minimum} | {values['measured']} / {values['eligible']} | {values['unavailable']} | {values['below_minimum']} | {values['not_applicable']} |"
            )
    lines.extend([
            "", "## Results by question category", "", "| Category | Planned | Passed | Failed | Errors | Missing |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for category, values in report["categories"].items():
        lines.append(
                f"| {category} | {values['planned']} | {values['passed']} | {values['failed']} | {values['errors']} | {values['missing']} |"
        )
    lines.extend([
            "", "## Judge control check", "",
            f"Status: {report['calibration'].get('status', 'unavailable')}. This is a small hand-labelled faithfulness sanity check, not statistical calibration of all four metrics.",
            "", "## Case diagnostics", ""])
    for row in report["results"]:
        lines.extend([f"### {row['case_id']} / {row['model']}", "", f"Status: {row_status(row)}", ""])
        if "answer_request_seconds" in row:
            lines.append(f"Answer request: {row['answer_request_seconds']:.3f} seconds.\n")
        if row.get("error"):
            lines.extend([str(row["error"]), ""])
        for d in row.get("dimensions", []):
            detail = d.get("error", d.get("reason", str(d.get("value", ""))))
            lines.append(f"- {d['name']}: {d['status']}. {detail}")
        lines.append("")
    return "\n".join(lines) + "\n"


def review_worksheet(report: dict) -> dict:
    return {
            "schema_version":
            1,
            "status":
            "pending_human_review",
            "instruction":
            "Review answers against references and captured contexts. Record acceptable/incorrect/incomplete/unsupported and judge disagreements before proposing thresholds. Keep this run's evidence immutable; save a separate reviewed copy.",
            "rows": [{
            "case_id":
            r["case_id"],
            "model":
            r["model"],
            "question":
            r.get("question"),
            "reference":
            r.get("reference"),
            "answer":
            r.get("answer"),
            "human_verdict":
            None,
            "review_flags": ["semantic_disagreement_candidate"]
            if not r.get("error") and all(d["status"] == "passed" for d in r.get("dimensions", [])[:2])
            and any(d.get("metric") and d["status"] in ("failed", "error") for d in r.get("dimensions", [])) else [],
            "judge_disagreements": [],
            "reviewer":
            None,
            "notes":
            None} for r in report["results"]]}


@step("Check: complete policy benchmark meets its experimental acceptance criteria")
def present_benchmark_report(report: dict) -> None:
    set_metadata(feature="RAG quality", story="Dataset benchmark")
    attach_text(json.dumps(report, indent=2), name="Benchmark results and case diagnostics")
    if "summary" in report:
        attach_text(markdown(report), name="Benchmark summary")
    for row in report.get("results", []):
        try:
            present_case_result(row)
        except AssertionError:
            # Preserve every independent case; the aggregate assertion fails afterwards.
            continue
    assertions.assert_benchmark_report(report)


@step("Check: benchmark case satisfies answer and semantic acceptance criteria")
def present_case_result(row: dict) -> None:
    attach_text(json.dumps(row, indent=2), name=f"{row['case_id']} / {row['model']}")
    assertions.assert_benchmark_case(row)
