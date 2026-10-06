"""Compare retained performance batches without confusing different workloads."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.performance.runner import validate_batch


def compare_batches(
    baseline: dict[str, Any], current: dict[str, Any], *, maximum_growth: float = 0.2
) -> dict[str, Any]:
    validate_batch(baseline)
    validate_batch(current)
    if (
        type(maximum_growth) not in (int, float)
        or not math.isfinite(maximum_growth)
        or not 0 <= maximum_growth <= 1
    ):
        raise ValueError("Latency growth must be a finite fraction between zero and one")
    fields = ("workload", "system", "python", "base_url", "warmup_requests")
    differences = [
        k
        for k in fields
        if not baseline.get("metadata", {}).get(k) == current.get("metadata", {}).get(k)
        or k not in baseline.get("metadata", {})
    ]
    for field in ("requests", "users"):
        if baseline[field] != current[field]:
            differences.append(field)
    for field in ("policy_sha256", "golden_dataset_sha256", "case_id"):
        if baseline.get("metadata", {}).get(field) != current.get("metadata", {}).get(field):
            differences.append(field)

    def configuration(row):
        return {
            k: v
            for k, v in row.get("metadata", {}).get("configuration", {}).items()
            if k != "chatModel"
        }

    if configuration(baseline) != configuration(current):
        differences.append("configuration")
    base = baseline["latency_seconds"]["p95"]
    if base <= 0 or baseline["failed"]:
        differences.append("baseline_not_healthy")
    growth = current["latency_seconds"]["p95"] / base - 1 if base > 0 else None
    status = (
        "incomparable"
        if differences
        else "regression"
        if current["failed"] or growth > maximum_growth
        else "passed"
    )
    return {
        "schema_version": 1,
        "status": status,
        "maximum_growth": maximum_growth,
        "p95_growth": growth,
        "baseline_p95": base,
        "current_p95": current["latency_seconds"]["p95"],
        "incomparable_fields": differences,
        "model_changed": baseline.get("metadata", {}).get("model_digest")
        != current.get("metadata", {}).get("model_digest"),
        "interpretation": "Bounded performance regression signal for comparable workload/machine metadata, not a statistically significant capacity estimate",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--maximum-growth", type=float, default=0.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    report = compare_batches(
        json.loads(args.baseline.read_text()),
        json.loads(args.current.read_text()),
        maximum_growth=args.maximum_growth,
    )
    write_sample(args.output, report)
    print(f"Performance comparison: {report['status']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
