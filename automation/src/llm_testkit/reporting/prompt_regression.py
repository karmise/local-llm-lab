"""Compare a declared prompt/case/model matrix without hiding missing or failed runs."""

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

from llm_testkit.core.provenance import normalize_configuration
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.datasets.prompts import load_prompt_catalog
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.junit import read_junit


def compare_prompts(
    report_path: Path,
    *,
    catalog_path: Path,
    dataset_path: Path,
    policy_file: Path,
    case_ids: list[str],
    models: list[str],
    candidate: str,
    repeat: int = 1,
) -> dict[str, Any]:
    catalog = load_prompt_catalog(catalog_path)
    dataset = load_golden_dataset(dataset_path, policy_file)
    variants = {v.id: v for v in catalog.variants}
    if candidate not in variants or candidate == catalog.baseline:
        raise ValueError("Select a candidate different from baseline")
    if not case_ids or not models or type(repeat) is not int or repeat < 1:
        raise ValueError("Declare nonempty cases/models and positive repetitions")
    if len(set(case_ids)) != len(case_ids) or len(set(models)) != len(models):
        raise ValueError("Duplicate case/model selection")
    if not set(case_ids) <= {c.id for c in dataset.cases}:
        raise ValueError("Unknown golden case")
    expected = set(product(case_ids, models, range(1, repeat + 1), (catalog.baseline, candidate)))
    junit = read_junit(report_path)
    entries = {
        identity: entry
        for identity, entry in junit.cases.items()
        if entry.classname.endswith("test_prompt_regression")
    }
    errors = []
    observed = {}
    for identity, entry in entries.items():
        p = entry.properties
        if entry.conflicts:
            errors.append(f"{identity}: Conflicting metadata: {sorted(entry.conflicts)}")
            continue
        try:
            key = (
                p["golden_case_id"],
                p["generation_model"],
                int(p["rag_iteration"]),
                p["prompt_id"],
            )
            if key not in expected or key in observed:
                raise ValueError("Unexpected or duplicated matrix run")
            variant = variants[p["prompt_id"]]
            if (
                p["golden_dataset_sha256"] != dataset.sha256
                or p["policy_sha256"] != dataset.policy_sha256
                or p["prompt_catalog_sha256"] != catalog.sha256
                or p["prompt_sha256"] != variant.sha256
                or p["prompt_version"] != variant.version
                or not p["model_digest"]
                or not p["thinking_mode"]
            ):
                raise ValueError("Changed or missing provenance")
            configuration = normalize_configuration(json.loads(p["workspace_configuration"]))
            prompt = configuration.pop("openAiPrompt")
            if prompt != variant.prompt or configuration["chatModel"] != key[1]:
                raise ValueError("Configuration does not match selected prompt/model")
            fingerprint = (
                p["model_digest"],
                p["thinking_mode"],
                json.dumps(configuration, sort_keys=True),
            )
            observed[key] = {"fingerprint": fingerprint, "outcome": entry.status}
        except (KeyError, ValueError, TypeError) as error:
            errors.append(f"{identity}: {error}")
    missing = sorted(expected - observed.keys())
    comparisons = []
    for case, model, iteration in product(case_ids, models, range(1, repeat + 1)):
        base = observed.get((case, model, iteration, catalog.baseline))
        new = observed.get((case, model, iteration, candidate))
        outcome = "incomplete"
        if base and new:
            if base["fingerprint"] != new["fingerprint"]:
                errors.append(
                    f"{case}/{model}/{iteration}: model or retrieval configuration changed"
                )
            elif base["outcome"] in ("error", "skipped") or new["outcome"] in ("error", "skipped"):
                pass
            elif base["outcome"] == "failed":
                outcome = "baseline_failed"
            else:
                outcome = "regression" if new["outcome"] == "failed" else "passed"
        comparisons.append(
            {
                "case": case,
                "model": model,
                "iteration": iteration,
                "outcome": outcome,
                "baseline": base["outcome"] if base else "missing",
                "candidate": new["outcome"] if new else "missing",
            }
        )
    outcomes = {c["outcome"] for c in comparisons}
    status = (
        "incomplete"
        if errors or missing or "incomplete" in outcomes
        else "regression"
        if "regression" in outcomes
        else "baseline_failed"
        if "baseline_failed" in outcomes
        else "passed"
    )
    return {
        "schema_version": 1,
        "status": status,
        "baseline": catalog.baseline,
        "candidate": candidate,
        "catalog_sha256": catalog.sha256,
        "golden_dataset_sha256": dataset.sha256,
        "report_sha256": junit.sha256,
        "scope": {
            "cases": case_ids,
            "models": models,
            "repeat": repeat,
            "expected_runs": len(expected),
        },
        "missing_runs": missing,
        "errors": errors,
        "comparisons": comparisons,
        "interpretation": "Acceptance regression within the declared scope; independent repetitions are not retries or statistical proof",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--case", action="append", required=True, dest="cases")
    parser.add_argument("--model", action="append", required=True, dest="models")
    parser.add_argument("--candidate", default="grounded_v2")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--catalog", type=Path, default=Path("test_data/prompt-variants.json"))
    parser.add_argument("--dataset", type=Path, default=Path("test_data/golden-policy.json"))
    parser.add_argument("--policy", type=Path, default=Path("test_data/company-policy.txt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    report = compare_prompts(
        args.report,
        catalog_path=args.catalog,
        dataset_path=args.dataset,
        policy_file=args.policy,
        case_ids=args.cases,
        models=args.models,
        candidate=args.candidate,
        repeat=args.repeat,
    )
    write_sample(args.output, report)
    print(
        f"Prompt comparison: {report['status']}; expected runs: {report['scope']['expected_runs']}"
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
