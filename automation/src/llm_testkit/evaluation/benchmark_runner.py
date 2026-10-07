"""Run a bounded, serial golden matrix with actual context and local RAGAS evidence."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from llm_testkit import assertions
from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.config import Settings
from llm_testkit.core.http_client import HttpClient
from llm_testkit.datasets.benchmark import CONTROL_IDS, make_plan, manifest
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.evaluation.benchmark import check_sample, evaluate_case
from llm_testkit.evaluation.calibration import evaluate_controls, load_controls, select_controls
from llm_testkit.evaluation.faithfulness import load_sample
from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.benchmark import markdown, review_worksheet, summarize
from llm_testkit.reporting.gates import load_quality_gates
from llm_testkit.reporting.junit import read_junit


def generate_sample(root: Path, directory: Path, case_id: str, model: str) -> dict:
    """Reuse existing setup, capture, acceptance and verified teardown fixtures."""
    node = f"tests/test_golden_rag.py::test_golden_policy_answer[{case_id}-{model}]"
    command = [
        sys.executable,
        "-m",
        "pytest",
        node,
        "--run-golden",
        "--capture-rag",
        "--rag-model",
        model,
        "--rag-repeat",
        "1",
        "--junitxml",
        str(directory / "generation.xml"),
        "-o",
        "addopts=-ra --strict-markers --strict-config --import-mode=importlib",
        "-q",
    ]
    environment = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTEST_ADDOPTS": ""}
    with (directory / "generation.log").open("x", encoding="utf-8") as log:
        process = subprocess.run(
            command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    junit = read_junit(directory / "generation.xml")
    if len(junit.cases) != 1:
        raise ValueError("Generation must produce exactly one JUnit case")
    result = next(iter(junit.cases.values()))
    if result.name != node.split("::", 1)[1] or result.conflicts:
        raise ValueError("Generation JUnit identity/metadata mismatch")
    if process.returncode not in (0, 1) or (process.returncode == 0) != (result.status == "passed"):
        raise ValueError("Generation exit code disagrees with its JUnit outcome")
    sample_path = Path(result.properties.get("evaluation_sample", ""))
    allowed = (root / "reports/rag-samples").resolve()
    if sample_path.resolve().parent != allowed or not sample_path.is_file():
        raise ValueError(f"Generation has no captured sample; JUnit status={result.status}")
    sample, _ = load_sample(sample_path)
    write_sample(directory / "sample.json", sample)
    return {
        "generation_status": result.status,
        "generation_exit_code": process.returncode,
        "generation_junit_sha256": junit.sha256,
        "generation_details": result.details,
    }


def calibrate(
    sample_path: Path, controls: Path, settings: Settings, model: str, digest: str
) -> dict:
    from llm_testkit.evaluation.ollama_judge import OllamaJudge

    report = {
        "status": "error",
        "judge_model": model,
        "judge_model_digest": digest,
        "controls_sha256": hashlib.sha256(controls.read_bytes()).hexdigest(),
        "control_ids": list(CONTROL_IDS),
        "results": [],
    }
    try:
        sample, report["sample_sha256"] = load_sample(sample_path)
        cases, _ = load_controls(controls, sample["retrieved_contexts"])
        cases = select_controls(cases, list(CONTROL_IDS))
        # Preserve declared order, independently of catalog order.
        cases = sorted(cases, key=lambda c: CONTROL_IDS.index(c["id"]))
        with HttpClient(settings.ollama_base_url, settings.http_timeout) as http:
            client = OllamaClient(http)
            report["results"] = asyncio.run(
                evaluate_controls(
                    sample, cases, lambda: OllamaJudge(client, model, settings.llm_timeout)
                )
            )
        statuses = {r["status"] for r in report["results"]}
        report["status"] = (
            "error" if "error" in statuses else "mismatch" if "mismatch" in statuses else "matched"
        )
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report


def run(root: Path, output: Path, plan, dataset, gates: dict, *, notify=print) -> dict:
    controls = root / "test_data/faithfulness-controls.json"
    definition = manifest(plan, dataset, gates, controls)
    definition["created_at"] = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True, exist_ok=False)
    write_sample(output / "manifest.json", definition)
    # Keep the exact reviewed inputs beside the run, not only paths into a mutable checkout.
    for name in ("golden-policy.json", "quality-gates.json", "faithfulness-controls.json"):
        (output / name).write_bytes((root / "test_data" / name).read_bytes())
    (output / "company-policy.txt").write_bytes(
        (root / "test_data/company-policy.txt").read_bytes()
    )
    settings = Settings.from_env(root.parent / ".runtime/anythingllm-api-key")
    rows = []
    calibration = {
        "status": "error",
        "error": "No paid-leave sample was available for judge controls",
    }
    digests = {}
    try:
        with HttpClient(settings.ollama_base_url, settings.http_timeout) as http:
            catalog = OllamaClient(http).list_models()
            assertions.assert_status_code(catalog, 200, context="Benchmark model catalog")
            for model in (*plan.models, plan.judge_model):
                digests[model] = assertions.assert_model_available(catalog.json()["models"], model)
    except Exception as error:
        calibration["error"] = f"Model preflight failed: {type(error).__name__}: {error}"
    cases = {case.id: case for case in dataset.cases}
    # Anchor judge controls before processing the rest of the declared matrix.
    expected_rows = sorted(definition["expected_rows"], key=lambda r: r["case_id"] != "paid_leave")
    for index, planned in enumerate(expected_rows, 1):
        row = dict(planned)
        directory = output / f"case-{index:03d}"
        directory.mkdir()
        notify(f"[{index}/{len(expected_rows)}] {row['case_id']} / {row['model']}", flush=True)
        try:
            if plan.judge_model not in digests or row["model"] not in digests:
                raise ValueError("Model preflight did not complete")
            generation = generate_sample(root, directory, row["case_id"], row["model"])
            row.update(generation)
            sample = check_sample(
                directory / "sample.json", dataset, cases[row["case_id"]], row["model"]
            )
            if sample["metadata"]["model_digest"] != digests[row["model"]]:
                raise ValueError("Generation weights changed after preflight")
            if row["case_id"] == "paid_leave" and "results" not in calibration:
                calibration = calibrate(
                    directory / "sample.json",
                    controls,
                    settings,
                    plan.judge_model,
                    digests[plan.judge_model],
                )
                write_sample(output / "judge-controls.json", calibration)
            row.update(
                evaluate_case(
                    directory / "sample.json",
                    directory=directory,
                    dataset=dataset,
                    dataset_path=root / "test_data/golden-policy.json",
                    policy_file=root / "test_data/company-policy.txt",
                    case=cases[row["case_id"]],
                    model=row["model"],
                    judge_model=plan.judge_model,
                    judge_digest=digests[plan.judge_model],
                    settings=settings,
                    minima=gates["minimum_scores"],
                )
            )
            row.update(generation)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["artifact_directory"] = directory.name
        write_sample(directory / "result.json", row)
        rows.append(row)
    try:
        report = summarize(definition, rows, calibration)
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "error",
            "manifest": definition,
            "results": rows,
            "calibration": calibration,
            "error": f"Aggregation rejected inconsistent evidence: {type(error).__name__}: {error}",
        }
    write_sample(output / "benchmark.json", report)
    if "summary" in report:
        (output / "benchmark.md").write_text(markdown(report), encoding="utf-8")
        write_sample(output / "human-review.json", review_worksheet(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Automation directory")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--max-model-calls", type=int, default=50)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without network/model calls"
    )
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    try:
        if output.exists():
            raise ValueError("Output already exists; choose a new directory")
        if (root / "tests/test_golden_rag.py").is_file() is False:
            raise ValueError("Run from automation or provide --root")
        dataset = load_golden_dataset(
            root / "test_data/golden-policy.json", root / "test_data/company-policy.txt"
        )
        gates = load_quality_gates(root / "test_data/quality-gates.json")
        plan = make_plan(
            dataset,
            case_ids=args.case_ids,
            models=args.models,
            judge_model=args.judge_model,
            max_model_calls=args.max_model_calls,
        )
        if args.dry_run:
            print(
                json.dumps(
                    manifest(plan, dataset, gates, root / "test_data/faithfulness-controls.json"),
                    indent=2,
                )
            )
            return 0
        if importlib.util.find_spec("ragas") is None:
            raise ValueError("Install requirements-evaluation.lock before running a benchmark")
        print(
            f"Serial benchmark: {len(plan.case_ids) * len(plan.models)} generations; at most {plan.maximum_calls} generation/judge calls; no retries.",
            flush=True,
        )
        report = run(root, output, plan, dataset, gates)
    except (ValueError, OSError, AssertionError) as error:
        parser.error(str(error))
    print(f"Benchmark: {report['status']}; evidence: {output}")
    return 0 if report["status"] == "checks_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
