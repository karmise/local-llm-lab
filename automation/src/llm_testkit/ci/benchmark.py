"""Use reviewed CI presets and bind saved evidence to the exact tested checkout."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from llm_testkit.datasets.benchmark import DEFAULT_CASES, make_plan, manifest
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.evaluation.benchmark_runner import run
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.qualification.plan import framework_checksum
from llm_testkit.reporting.benchmark_evidence import load_saved_benchmark
from llm_testkit.reporting.gates import load_quality_gates

PROFILES = {"smoke": ("paid_leave", "gym_missing"), "curated": DEFAULT_CASES}
MODELS = {"primary": ("qwen3.5:4b", ), "comparison": ("qwen3.5:4b", "qwen2.5:7b")}
BASELINES = ("golden-policy.json", "company-policy.txt", "quality-gates.json", "faithfulness-controls.json")


def inputs(root: Path, profile: str, models: str):
    if profile not in PROFILES or models not in MODELS:
        raise ValueError("Select a reviewed CI profile and model preset")
    dataset = load_golden_dataset(root / "test_data/golden-policy.json", root / "test_data/company-policy.txt")
    gates = load_quality_gates(root / "test_data/quality-gates.json")
    plan = make_plan(dataset, case_ids=list(PROFILES[profile]), models=list(MODELS[models]), max_model_calls=80)
    return dataset, gates, plan


def execution_context(root: Path, revision: str, profile: str, models: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("CI evidence requires the full tested commit SHA")
    return {
            "schema_version": 1,
            "revision": revision,
            "profile": profile,
            "models": models,
            "framework_source_sha256": framework_checksum(root)}


def validate_ci_benchmark(root: Path, path: Path, *, revision: str, profile: str, models: str) -> dict:
    dataset, gates, plan = inputs(root, profile, models)
    directory = path.parent
    context = execution_context(root, revision, profile, models)
    if json.loads((directory / "ci-context.json").read_text()) != context:
        raise ValueError("CI context differs from the tested revision, source or requested scope")
    for filename in BASELINES:
        if (directory / filename).read_bytes() != (root / "test_data" / filename).read_bytes():
            raise ValueError(f"CI baseline differs from the tested checkout: {filename}")
    definition = json.loads((directory / "manifest.json").read_text())
    definition.pop("created_at", None)
    if definition != manifest(plan, dataset, gates, root / "test_data/faithfulness-controls.json"):
        raise ValueError("CI manifest differs from the complete requested plan")
    report = load_saved_benchmark(path)
    weights = json.loads((root.parent / "config/ci-models.json").read_text())
    calibration = report["calibration"]
    if calibration.get("judge_model_digest") not in (None, weights[plan.judge_model]):
        raise ValueError("CI judge weights differ from the reviewed model lock")
    for row in report["results"]:
        if row.get("error"):
            continue  # The aggregate remains an error; missing evidence never becomes success.
        sample = json.loads((directory / row["artifact_directory"] / "sample.json").read_text())
        metadata = sample["metadata"]
        node = f"tests/test_golden_rag.py::test_golden_policy_answer[{row['case_id']}-{row['model']}]"
        source = hashlib.sha256((root / "tests/test_golden_rag.py").read_bytes()).hexdigest()
        if (metadata.get("framework_source_sha256") != context["framework_source_sha256"]
                    or metadata.get("test_source_sha256") != source or metadata.get("test_node_id") != node):
            raise ValueError("CI sample was not produced by the tested framework and golden test")
        if row.get("model_digest") != weights[row["model"]]:
            raise ValueError("CI generation weights differ from the reviewed model lock")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("plan", "produce"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--profile", choices=PROFILES, default="curated")
    parser.add_argument("--models", choices=MODELS, default="primary")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    context = execution_context(root, args.revision, args.profile, args.models)
    dataset, gates, plan = inputs(root, args.profile, args.models)
    if args.operation == "plan":
        write_sample(
                args.output, {
                "context": context,
                "manifest": manifest(plan, dataset, gates, root / "test_data/faithfulness-controls.json")})
        return 0
    report = run(root, args.output.resolve(), plan, dataset, gates)
    write_sample(args.output / "ci-context.json", context)
    print(f"CI benchmark: {report['status']}; maximum generation/judge calls: {plan.maximum_calls}", flush=True)
    return 0 if report["status"] == "checks_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
