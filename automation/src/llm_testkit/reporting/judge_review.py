"""Expose judge/reference disagreement candidates from verified saved benchmarks."""

import argparse
import hashlib
import json
from pathlib import Path

from llm_testkit.evaluation.benchmark import EVIDENCE_METRICS
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.benchmark_evidence import load_saved_benchmark


def review_benchmark(path: Path) -> dict:
    benchmark = load_saved_benchmark(path)
    root = path.parent
    rows = []
    for case in benchmark["results"]:
        failures = [
                dimension for dimension in case.get("dimensions", [])
                if dimension.get("metric") and dimension["status"] in {"failed", "error"}]
        evidence = {}
        for name, metrics in EVIDENCE_METRICS.items():
            if any(dimension.get("metric") in metrics for dimension in failures):
                source = root / case["artifact_directory"] / f"{name}.json"
                raw = source.read_bytes()
                original = json.loads(raw)
                evidence[name] = {
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "result": original.get("result"),
                        "error": original.get("error"),
                        "judge_model": original.get("judge_model"),
                        "judge_model_digest": original.get("judge_model_digest")}
        deterministic = case.get("dimensions", [])[:2]
        rows.append({
                "case_id":
                case["case_id"],
                "model":
                case["model"],
                "sample_sha256":
                case.get("sample_sha256"),
                "question":
                case.get("question"),
                "reference":
                case.get("reference"),
                "answer":
                case.get("answer"),
                "generation_status":
                case.get("generation_status"),
                "deterministic_checks":
                deterministic,
                "failed_semantic_dimensions":
                failures,
                "assessment":
                "judge_reference_disagreement_candidate" if failures and len(deterministic) == 2
                and all(dimension["status"] == "passed" for dimension in deterministic)
                and case.get("generation_status") == "passed" and not case.get("error") else "no_automatic_assessment",
                "evidence":
                evidence,
                "human_verdict":
                None})
    return {
            "schema_version":
            1,
            "benchmark_sha256":
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "policy_sha256":
            benchmark["manifest"]["policy_sha256"],
            "golden_dataset_sha256":
            benchmark["manifest"]["golden_dataset_sha256"],
            "original_benchmark_status":
            benchmark["status"],
            "human_review":
            "pending",
            "threshold_decision":
            "retain_experimental_thresholds",
            "rows":
            rows,
            "interpretation":
            "Passing fact/source rules do not prove answer correctness. Candidates need contextual "
            "review; this report does not override failures, approve a judge or change any source artifact."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new file")
    try:
        report = review_benchmark(args.benchmark)
    except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
        parser.error(f"Cannot review invalid benchmark evidence: {error}")
    write_sample(args.output, report)
    count = sum(row["assessment"] == "judge_reference_disagreement_candidate" for row in report["rows"])
    print(
            f"Verified benchmark: {report['original_benchmark_status']}; disagreement candidates: {count}; "
            f"human review pending. Saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
