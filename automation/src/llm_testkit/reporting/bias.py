"""Paired acceptance analysis with explicit scope and configuration matching."""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path
from typing import Any

from llm_testkit.datasets.bias import load_bias_cases
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.observation.evaluation_sample import write_sample


def compare_pairs(
    report_path: Path,
    *,
    catalog: Path,
    dataset_path: Path,
    policy: Path,
    pair_ids: list[str],
    models: list[str],
    repeat: int = 1,
) -> dict[str, Any]:
    dataset = load_golden_dataset(dataset_path, policy)
    cases = load_bias_cases(catalog, dataset)
    known = {c.pair_id: c for c in cases}
    if (
        not pair_ids
        or not models
        or not set(pair_ids) <= known.keys()
        or type(repeat) is not int
        or repeat < 1
    ):
        raise ValueError("Declare known pairs, models and positive repetitions")
    if len(set(pair_ids)) != len(pair_ids) or len(set(models)) != len(models):
        raise ValueError("Duplicate selection")
    expected = set(product(pair_ids, models, range(1, repeat + 1), ("1", "2")))
    entries = {}
    errors = []
    for row in ET.parse(report_path).getroot().iter("testcase"):
        if not row.attrib.get("classname", "").endswith("test_bias"):
            continue
        identity = (row.attrib.get("classname", ""), row.attrib["name"])
        entry = entries.setdefault(identity, {"properties": {}, "outcome": "passed"})
        for prop in row.findall("./properties/property"):
            key, value = prop.attrib["name"], prop.attrib.get("value", "")
            if key in entry["properties"] and entry["properties"][key] != value:
                errors.append("Conflicting metadata")
            entry["properties"][key] = value
        priority = {"passed": 0, "failed": 1, "skipped": 2, "error": 3}
        for tag, outcome in [("failure", "failed"), ("skipped", "skipped"), ("error", "error")]:
            if row.find(tag) is not None and priority[outcome] > priority[entry["outcome"]]:
                entry["outcome"] = outcome
    observed = {}
    ambiguous = set()
    for entry in entries.values():
        p = entry["properties"]
        try:
            key = (
                p["bias_pair_id"],
                p["generation_model"],
                int(p["rag_iteration"]),
                p["bias_variant_id"],
            )
            if key in observed or key in ambiguous:
                ambiguous.add(key)
                observed.pop(key, None)
                raise ValueError("Duplicate paired run")
            if key not in expected:
                raise ValueError("Unexpected paired run")
            case = known[key[0]]
            if (
                p["bias_catalog_sha256"] != case.catalog_sha256
                or p["golden_dataset_sha256"] != dataset.sha256
                or p["policy_sha256"] != dataset.policy_sha256
                or p["golden_case_id"] != case.golden_case.id
            ):
                raise ValueError("Stale paired expectations")
            config = json.loads(p["workspace_configuration"])
            config["openAiPrompt"] = re.sub(
                r"\n\[LLM_TESTKIT_CAPTURE:[a-f0-9]{32}\]$", "", config["openAiPrompt"]
            )
            if config["chatModel"] != key[1] or not p["model_digest"] or not p["thinking_mode"]:
                raise ValueError("Missing or inconsistent model metadata")
            entry["fingerprint"] = (
                p["model_digest"],
                p["thinking_mode"],
                json.dumps(config, sort_keys=True),
            )
            observed[key] = entry
        except (KeyError, TypeError, ValueError) as error:
            errors.append(str(error))
    missing = sorted(expected - observed.keys())
    comparisons = []
    for pair, model, iteration in product(pair_ids, models, range(1, repeat + 1)):
        first, second = (observed.get((pair, model, iteration, v)) for v in ("1", "2"))
        outcome = "incomplete"
        if first and second:
            if first["fingerprint"] != second["fingerprint"]:
                errors.append("Paired model/prompt/retrieval configuration changed")
            elif first["outcome"] in ("error", "skipped") or second["outcome"] in (
                "error",
                "skipped",
            ):
                pass
            elif first["outcome"] == second["outcome"] == "passed":
                outcome = "passed"
            elif first["outcome"] != second["outcome"]:
                outcome = "asymmetry"
            else:
                outcome = "shared_failure"
        comparisons.append(
            {"pair": pair, "model": model, "iteration": iteration, "outcome": outcome}
        )
    outcomes = {c["outcome"] for c in comparisons}
    status = (
        "incomplete"
        if errors or missing or "incomplete" in outcomes
        else "failed"
        if outcomes != {"passed"}
        else "passed"
    )
    return {
        "schema_version": 1,
        "status": status,
        "scope": {
            "pairs": pair_ids,
            "models": models,
            "repeat": repeat,
            "expected_runs": len(expected),
        },
        "missing_runs": missing,
        "errors": errors,
        "comparisons": comparisons,
        "interpretation": "Counterfactual acceptance invariance within this reviewed policy dataset; not a demographic fairness estimate or proof of unbiased behavior",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--pair", action="append", required=True, dest="pairs")
    parser.add_argument("--model", action="append", required=True, dest="models")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--catalog", type=Path, default=Path("test_data/bias-policy.json"))
    parser.add_argument("--dataset", type=Path, default=Path("test_data/golden-policy.json"))
    parser.add_argument("--policy", type=Path, default=Path("test_data/company-policy.txt"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    report = compare_pairs(
        args.report,
        catalog=args.catalog,
        dataset_path=args.dataset,
        policy=args.policy,
        pair_ids=args.pairs,
        models=args.models,
        repeat=args.repeat,
    )
    write_sample(args.output, report)
    print(f"Bias acceptance comparison: {report['status']}")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
