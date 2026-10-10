"""Run bounded judge controls without generating application answers or changing gates."""

import argparse
import asyncio
import math
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

from llm_testkit import assertions
from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.config import Settings
from llm_testkit.core.http_client import HttpClient
from llm_testkit.core.scores import require_score
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.datasets.judge_controls import load_judge_controls, maximum_calls, select_judge_controls
from llm_testkit.evaluation.correctness import result_from_calls, score_correctness
from llm_testkit.evaluation.faithfulness import score_sample, validate_result
from llm_testkit.observation.evaluation_sample import write_sample


def result_from_evidence(control: dict[str, Any], calls: list[dict[str, Any]], value: float) -> dict[str, Any]:
    require_score(value, "Judge control score")
    if len(calls) != maximum_calls(control):
        raise ValueError("Judge control has an unexpected number of calls")
    metric = control["metric"]
    if metric == "faithfulness":
        result = validate_result({
                "value": value,
                "statements": calls[0]["output"]["statements"],
                "verdicts": calls[1]["output"]["statements"]})
        return {"value": result["value"], "response": result["verdicts"]}
    if metric == "factual_correctness":
        result = result_from_calls(calls, value)
        return {
                "value": result["value"],
                "response": result["response_verdicts"],
                "reference": result["reference_verdicts"]}
    if metric == "context_precision":
        rows = [call["output"] for call in calls]
        values = [row["verdict"] for row in rows]
        expected = sum(sum(values[:i + 1]) / (i + 1) * v for i, v in enumerate(values)) / (sum(values) + 1e-10)
        result = {"value": value, "verdicts": values}
    else:
        rows = calls[0]["output"]["classifications"]
        values = [row["attributed"] for row in rows]
        if not rows or len({row["statement"] for row in rows}) != len(rows):
            raise ValueError("Recall requires nonempty, unique claim evidence")
        expected = sum(values) / len(values)
        result = {"value": value, "reference": [{**row, "verdict": row["attributed"]} for row in rows]}
    if (any(type(v) is not int or v not in (0, 1) for v in values)
                or any(not isinstance(row.get("reason"), str) or not row["reason"].strip() for row in rows)):
        raise ValueError("Judge verdicts must be binary and include reasons")
    if not math.isclose(value, expected, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Judge score disagrees with raw verdicts")
    return result


def label_mismatches(result: dict[str, Any], control: dict[str, Any]) -> list[str]:
    mismatches = []
    if not math.isclose(result["value"], control["expected_score"], rel_tol=0, abs_tol=1e-8):
        mismatches.append(f"Expected score {control['expected_score']}, observed {result['value']}")
    for field, labels in control["labels"].items():
        if field == "verdicts":
            if result[field] != labels:
                mismatches.append("Ordered context verdicts differ from labels")
            continue
        rows = result[field]
        covered = set()
        for rule in labels:
            hits = {i for i, row in enumerate(rows) if re.search(rule["pattern"], row["statement"], re.IGNORECASE)}
            if not hits or any(rows[i]["verdict"] != rule["verdict"] for i in hits):
                mismatches.append(f"{field} claim label disagrees: {rule['pattern']}")
            covered.update(hits)
        if covered != set(range(len(rows))):
            mismatches.append(f"Unlabelled {field} claims require review")
    return mismatches


async def score_control(control: dict[str, Any], judge: Any) -> float:
    from ragas.metrics.collections import ContextPrecisionWithReference, ContextRecall

    metric = control["metric"]
    if metric == "faithfulness":
        return (await score_sample(control, judge))["value"]
    if metric == "factual_correctness":
        return (await score_correctness(control, judge))["value"]
    scorer = ContextPrecisionWithReference(llm=judge) if metric == "context_precision" else ContextRecall(llm=judge)
    scored = await scorer.ascore(
            user_input=control["user_input"], reference=control["reference"],
            retrieved_contexts=control["retrieved_contexts"])
    return scored.value


async def evaluate_judge_controls(
        cases: list[dict[str, Any]], judge_factory: Callable[[int], Any],
        scorer: Callable = score_control) -> list[dict[str, Any]]:
    if not cases or sum(maximum_calls(case) for case in cases) > 32:
        raise ValueError("Judge validation requires a bounded nonempty batch")
    results = []
    for control in cases:
        row = {"id": control["id"], "metric": control["metric"], "control": control, "status": "error"}
        judge = None
        try:
            judge = judge_factory(maximum_calls(control))
            value = await scorer(control, judge)
            row["result"] = result_from_evidence(control, judge.calls, value)
            row["mismatches"] = label_mismatches(row["result"], control)
            row["status"] = "mismatch" if row["mismatches"] else "matched"
        except Exception as error:
            row["error"] = {"type": type(error).__name__, "message": str(error)}
        row["judge_calls"] = judge.calls if judge is not None else []
        results.append(row)
    return results


def summarize_validation(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results or any(row.get("status") not in {"matched", "mismatch", "error"} for row in results):
        raise ValueError("Judge validation requires nonempty results with known statuses")
    counts = {status: sum(row["status"] == status for row in results) for status in ("matched", "mismatch", "error")}
    return {
            "status": "error" if counts["error"] else "mismatch" if counts["mismatch"] else "matched",
            "counts": counts,
            "observed_judge_calls": sum(len(row["judge_calls"]) for row in results),
            "release_suitability": "not_established",
            "human_review": "pending",
            "threshold_decision": "retain_experimental_thresholds"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", type=Path, default=Path("test_data/judge-validation.json"))
    parser.add_argument("--dataset", type=Path, default=Path("test_data/golden-policy.json"))
    parser.add_argument("--policy", type=Path, default=Path("test_data/company-policy.txt"))
    parser.add_argument("--control", action="append", dest="identifiers")
    parser.add_argument("--max-judge-calls", type=int, default=18)
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.dry_run and args.output is None:
        parser.error("Live judge validation requires --output")
    if args.output is not None and args.output.exists():
        parser.error("Output already exists; choose a new file")
    try:
        dataset = load_golden_dataset(args.dataset, args.policy)
        catalog = load_judge_controls(args.controls, dataset, args.policy)
        cases = select_judge_controls(catalog, args.identifiers, args.max_judge_calls)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    planned_calls = sum(maximum_calls(case) for case in cases)
    print(f"Controls: {len(cases)}; maximum judge calls: {planned_calls}; generations: 0; concurrency: 1; retries: 0")
    if args.dry_run:
        return 0
    from llm_testkit.evaluation.ollama_judge import OllamaJudge

    report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response_origin": "synthetic_engineering_labelled_controls",
            "catalog_version": catalog.version,
            "catalog_sha256": catalog.sha256,
            "policy_sha256": catalog.policy_sha256,
            "golden_dataset_sha256": catalog.dataset_sha256,
            "ragas_version": version("ragas"),
            "judge_model": args.judge_model,
            "maximum_judge_calls": planned_calls,
            "execution": {
            "concurrency": 1,
            "retries": 0,
            "generations": 0},
            "scope": "Small judge suitability check; not statistical calibration or application acceptance",
            "status": "error",
            "results": []}
    settings = Settings.from_env()
    http = HttpClient(settings.ollama_base_url, settings.http_timeout)
    try:
        client = OllamaClient(http)
        inventory = client.list_models()
        assertions.assert_status_code(inventory, 200, context="Judge validation model inventory")
        report["judge_model_digest"] = assertions.assert_model_available(inventory.json()["models"], args.judge_model)

        def factory(budget):
            return OllamaJudge(client, args.judge_model, settings.llm_timeout, max_calls=budget)

        configuration = factory(1)
        report["judge_configuration"] = {"options": configuration.options, "think": False, "retries": 0}
        report["results"] = asyncio.run(evaluate_judge_controls(cases, factory))
        report.update(summarize_validation(report["results"]))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        http.close()
    write_sample(args.output, report)
    for row in report["results"]:
        print(f"{row['id']}: {row['status']}")
    print(f"Judge validation: {report['status']}; evidence saved to {args.output}")
    return 0 if report["status"] == "matched" else 1


if __name__ == "__main__":
    raise SystemExit(main())
