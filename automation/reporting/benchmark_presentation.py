"""Present a benchmark summary to people: Markdown, a review worksheet and Allure steps."""

import json

import assertions
from reporting.benchmark import row_status
from reporting.steps import attach_text, set_metadata, step


def markdown(report: dict) -> str:
    lines = [
            "# Policy quality benchmark", "", f"Status: **{report['status']}**", "", report["interpretation"], "",
            "## Results by generation model", "",
            "| Model | Planned | Passed | Failed | Errors | Missing | Pass rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model, summary in report["models"].items():
        lines.append(
                f"| {model} | {summary['planned']} | {summary['passed']} | {summary['failed']} | "
                f"{summary['errors']} | {summary['missing']} | {summary['pass_rate']:.1%} |")
    lines.extend([
            "", "## Answer request timing", "",
            (
            "Descriptive timings include retrieval, model loading and generation. They exclude "
            "fixture setup, judge calls and cleanup. One observation per case is not a latency SLO or"
            " a statistically reliable speed ranking."), "",
            "| Model | Measured / planned | Unavailable | Mean seconds | Min seconds | Max seconds |",
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
                    f"| {model} | {metric} | {avg} | {minimum} | {values['measured']} / "
                    f"{values['eligible']} | {values['unavailable']} | {values['below_minimum']} | "
                    f"{values['not_applicable']} |")
    lines.extend([
            "", "## Results by question category", "", "| Category | Planned | Passed | Failed | Errors | Missing |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for category, values in report["categories"].items():
        lines.append(
                f"| {category} | {values['planned']} | {values['passed']} | {values['failed']} | "
                f"{values['errors']} | {values['missing']} |")
    lines.extend([
            "", "## Judge control check", "",
            (
            f"Status: {report['calibration'].get('status', 'unavailable')}. This is a small "
            f"hand-labelled faithfulness sanity check, not statistical calibration of all four "
            f"metrics."), "", "## Case diagnostics", ""])
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
            "schema_version": 1,
            "status": "pending_human_review",
            "instruction": (
            "Review answers against references and captured contexts. Record "
            "acceptable/incorrect/incomplete/unsupported and judge disagreements before proposing "
            "thresholds. Keep this run's evidence immutable; save a separate reviewed copy."),
            "rows": [{
            "case_id": r["case_id"],
            "model": r["model"],
            "question": r.get("question"),
            "reference": r.get("reference"),
            "answer": r.get("answer"),
            "human_verdict": None,
            "review_flags": ["semantic_disagreement_candidate"]
            if not r.get("error") and all(d["status"] == "passed" for d in r.get("dimensions", [])[:2])
            and any(d.get("metric") and d["status"] in ("failed", "error") for d in r.get("dimensions", [])) else [],
            "judge_disagreements": [],
            "reviewer": None,
            "notes": None} for r in report["results"]]}


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
