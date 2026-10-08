"""Compare retained performance batches without confusing different workloads."""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

from llm_testkit.core.provenance import normalize_configuration
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.performance.runner import validate_batch


def compare_batches(baseline: dict[str, Any], current: dict[str, Any], *, maximum_growth: float = 0.2) -> dict[str,
        Any]:
    validate_batch(baseline)
    validate_batch(current)
    if (type(maximum_growth) not in (int, float) or not math.isfinite(maximum_growth) or not 0 <= maximum_growth <= 1):
        raise ValueError("Latency growth must be a finite fraction between zero and one")
    metadata = [row.get("metadata") for row in (baseline, current)]
    metadata = [m if isinstance(m, dict) else {} for m in metadata]
    before, after = metadata
    differences = []
    for field in ("workload", "system", "machine", "python", "base_url"):
        if any(not isinstance(m.get(field), str) or not m[field].strip()
                for m in metadata) or before.get(field) != after.get(field):
            differences.append(field)
    if any(m.get("workload") not in ("health", "rag") for m in metadata):
        differences.append("unsupported_workload")
    for field in ("warmup_requests", "timeout"):
        valid = all(
                type(m.get(field)) is int and m[field] >= 0 if field == "warmup_requests" else type(m.get(field)) in (
                int, float) and math.isfinite(m[field]) and m[field] > 0 for m in metadata)
        if not valid or before.get(field) != after.get(field):
            differences.append(field)
    for field in ("requests", "users"):
        if baseline[field] != current[field]:
            differences.append(field)
    if any(m.get("workload") == "rag" for m in metadata):
        for field in ("policy_sha256", "golden_dataset_sha256", "case_id"):
            valid = all(isinstance(m.get(field), str) and bool(m[field].strip()) for m in metadata)
            if field.endswith("sha256"):
                valid = valid and all(re.fullmatch(r"[a-f0-9]{64}", m[field]) for m in metadata)
            if not valid or before.get(field) != after.get(field):
                differences.append(field)
        for field in ("generation_model", "model_digest"):
            if any(not isinstance(m.get(field), str) or not m[field].strip() for m in metadata):
                differences.append(field)
        configurations = []
        for m in metadata:
            try:
                config = normalize_configuration(m.get("configuration"))
                if not config or config.get("chatModel") != m.get("generation_model"):
                    raise ValueError("Missing or inconsistent model configuration")
                configurations.append({k: v for k, v in config.items() if k != "chatModel"})
            except ValueError:
                differences.append("configuration")
        if len(configurations) == 2 and configurations[0] != configurations[1]:
            differences.append("configuration")
    differences = list(dict.fromkeys(differences))
    base = baseline["latency_seconds"]["p95"]
    if base <= 0 or baseline["failed"]:
        differences.append("baseline_not_healthy")
    growth = current["latency_seconds"]["p95"] / base - 1 if base > 0 else None
    status = (
            "incomparable"
            if differences else "regression" if current["failed"] or growth > maximum_growth + 1e-12 else "passed")
    return {
            "schema_version":
            1,
            "status":
            status,
            "maximum_growth":
            maximum_growth,
            "p95_growth":
            growth,
            "baseline_p95":
            base,
            "current_p95":
            current["latency_seconds"]["p95"],
            "incomparable_fields":
            differences,
            "model_changed": (before.get("generation_model"), before.get("model_digest"))
            != (after.get("generation_model"), after.get("model_digest")),
            "interpretation":
            "Bounded performance regression signal for comparable workload/machine metadata, not a statistically significant capacity estimate"
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
            json.loads(args.baseline.read_text()), json.loads(args.current.read_text()),
            maximum_growth=args.maximum_growth)
    write_sample(args.output, report)
    print(f"Performance comparison: {report['status']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
