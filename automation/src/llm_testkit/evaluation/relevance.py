"""Bounded reference-based context precision and recall on captured contexts."""

import argparse
import asyncio
import math
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llm_testkit import assertions
from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.config import Settings
from llm_testkit.core.http_client import HttpClient
from llm_testkit.datasets.golden import GoldenCase, GoldenDataset, load_golden_dataset
from llm_testkit.evaluation.correctness import bind_case
from llm_testkit.evaluation.faithfulness import load_sample
from llm_testkit.observation.evaluation_sample import write_sample

if TYPE_CHECKING:
    from llm_testkit.evaluation.ollama_judge import OllamaJudge


def validate_relevance(
    result: dict[str, Any], case: GoldenCase, context_count: int
) -> dict[str, Any]:
    verdicts = result["precision_verdicts"]
    classifications = result["recall_classifications"]
    if len(verdicts) != context_count or not 1 <= context_count <= 4:
        raise ValueError(
            "Context precision requires one verdict per captured context, maximum four"
        )
    if not classifications or len({c["statement"] for c in classifications}) != len(
        classifications
    ):
        raise ValueError("Recall classifications must be nonempty and unique")
    for row, key in [(v, "verdict") for v in verdicts] + [
        (c, "attributed") for c in classifications
    ]:
        if type(row[key]) is not int or row[key] not in (0, 1):
            raise ValueError("Relevance verdicts must be binary integers")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError("Relevance requires nonempty reasons")
    combined = " ".join(c["statement"] for c in classifications)
    for label, pattern in case.required_patterns:
        if not re.search(pattern, combined, re.IGNORECASE):
            raise ValueError(f"Recall classifications omit a labelled reference fact: {label}")
    values = [v["verdict"] for v in verdicts]
    expected_precision = sum(sum(values[: i + 1]) / (i + 1) * v for i, v in enumerate(values)) / (
        sum(values) + 1e-10
    )
    expected_recall = sum(c["attributed"] for c in classifications) / len(classifications)
    for key, expected in [
        ("context_precision", expected_precision),
        ("context_recall", expected_recall),
    ]:
        assertions.assert_quality_score(result[key])
        if not math.isclose(result[key], expected, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"{key} does not match its verdicts")
    return result


async def score_relevance(
    sample: dict[str, Any],
    case: GoldenCase,
    precision_judge: "OllamaJudge",
    recall_judge: "OllamaJudge",
) -> dict[str, Any]:
    from ragas.metrics.collections import ContextPrecisionWithReference, ContextRecall

    contexts = sample["retrieved_contexts"]
    if not 1 <= len(contexts) <= 4:
        raise ValueError("Evaluate one to four captured contexts; contexts are never truncated")
    precision = await ContextPrecisionWithReference(llm=precision_judge).ascore(
        user_input=sample["user_input"], reference=sample["reference"], retrieved_contexts=contexts
    )
    recall = await ContextRecall(llm=recall_judge).ascore(
        user_input=sample["user_input"], reference=sample["reference"], retrieved_contexts=contexts
    )
    if len(precision_judge.calls) != len(contexts) or len(recall_judge.calls) != 1:
        raise ValueError("Unexpected relevance judge call count")
    return validate_relevance(
        {
            "context_precision": precision.value,
            "context_recall": recall.value,
            "precision_verdicts": [c["output"] for c in precision_judge.calls],
            "recall_classifications": recall_judge.calls[0]["output"]["classifications"],
        },
        case,
        len(contexts),
    )


def evaluate_relevance_report(
    sample_path: Path,
    *,
    dataset_path: Path,
    policy_file: Path,
    case_id: str,
    settings: Settings,
    judge_model: str = "qwen3.5:4b",
) -> dict[str, Any]:
    from llm_testkit.evaluation.ollama_judge import OllamaJudge

    report = {
        "schema_version": 1,
        "metric": "context_relevance",
        "status": "error",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ragas_version": version("ragas"),
        "judge_model": judge_model,
        "threshold": None,
        "interpretation": "Reference-based average precision and claim recall; exploratory measurements",
    }
    http = None
    precision_judge = recall_judge = None
    try:
        sample, checksum = load_sample(sample_path)
        dataset = load_golden_dataset(dataset_path, policy_file)
        case = bind_case(sample, dataset, case_id)
        count = len(sample["retrieved_contexts"])
        if not 1 <= count <= 4:
            raise ValueError("Evaluate one to four captured contexts; contexts are never truncated")
        report.update(
            sample_sha256=checksum,
            golden_dataset_sha256=dataset.sha256,
            golden_case_id=case.id,
            reference=case.reference,
            question=case.question,
        )
        http = HttpClient(settings.ollama_base_url, settings.http_timeout)
        client = OllamaClient(http)
        catalog = client.list_models()
        assertions.assert_status_code(catalog, 200, context="Relevance judge catalog")
        report["judge_model_digest"] = assertions.assert_model_available(
            catalog.json()["models"], judge_model
        )
        precision_judge = OllamaJudge(client, judge_model, settings.llm_timeout, max_calls=count)
        recall_judge = OllamaJudge(client, judge_model, settings.llm_timeout, max_calls=1)
        report["judge_configuration"] = {
            "options": precision_judge.options,
            "think": False,
            "retries": 0,
            "maximum_calls": count + 1,
        }
        report["result"] = asyncio.run(score_relevance(sample, case, precision_judge, recall_judge))
        report["status"] = "completed"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        report["precision_calls"] = precision_judge.calls if precision_judge else []
        report["recall_calls"] = recall_judge.calls if recall_judge else []
        if http:
            http.close()
    return report


def check_relevance_evidence(
    evidence: dict[str, Any], checksum: str, sample: dict[str, Any], dataset: GoldenDataset
) -> dict[str, Any]:
    if (
        evidence.get("schema_version") != 1
        or evidence.get("metric") != "context_relevance"
        or evidence.get("status") != "completed"
    ):
        raise ValueError("Relevance requires completed context evidence")
    if (
        evidence.get("sample_sha256") != checksum
        or evidence.get("golden_dataset_sha256") != dataset.sha256
    ):
        raise ValueError("Relevance sample/dataset checksum mismatch")
    case = bind_case(sample, dataset, evidence["golden_case_id"])
    if evidence.get("question") != case.question or evidence.get("reference") != case.reference:
        raise ValueError("Relevance reference/question mismatch")
    result = validate_relevance(evidence["result"], case, len(sample["retrieved_contexts"]))
    if (
        len(evidence["recall_calls"]) != 1
        or len(evidence["precision_calls"]) != len(sample["retrieved_contexts"])
        or [c["output"] for c in evidence["precision_calls"]] != result["precision_verdicts"]
        or evidence["recall_calls"][0]["output"]["classifications"]
        != result["recall_classifications"]
    ):
        raise ValueError("Relevance summary differs from raw judge calls")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("--case", required=True, dest="case_id")
    parser.add_argument("--dataset", type=Path, default=Path("test_data/golden-policy.json"))
    parser.add_argument("--policy", type=Path, default=Path("test_data/company-policy.txt"))
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    report = evaluate_relevance_report(
        args.sample,
        dataset_path=args.dataset,
        policy_file=args.policy,
        case_id=args.case_id,
        settings=Settings.from_env(),
        judge_model=args.judge_model,
    )
    write_sample(args.output, report)
    print(f"Relevance: {report['status']}; evidence saved to {args.output}")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
