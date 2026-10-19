"""Evaluate all applicable dimensions without replacing judge errors with scores."""

import hashlib
import math
import re
from pathlib import Path
from typing import Any

import assertions
from config import Settings
from evaluation.correctness import bind_case, check_correctness_evidence, evaluate_correctness_report
from evaluation.faithfulness import evaluate_sample_report, load_sample
from evaluation.relevance import check_relevance_evidence, evaluate_relevance_report
from models.golden import GoldenCase, GoldenDataset
from observation.evaluation_sample import write_sample
from reporting.quality import check_faithfulness_evidence

EVIDENCE_METRICS = {
        "faithfulness": ("faithfulness", ),
        "correctness": ("factual_correctness", ),
        "relevance": ("context_precision", "context_recall")}
# The policy is uploaded under its own file name, so every run gives the model the same context; the test's own
# workspace, which holds only this document, is what ties the context to the test.
POLICY_DOCUMENT = "company-policy.txt"


def metric_dimensions(name, evidence, checksum, sample, dataset, judge_model, judge_digest, minima):
    """Use the same evidence validation for fresh runs and offline rendering."""
    try:
        if (evidence.get("judge_model") != judge_model or evidence.get("judge_model_digest") != judge_digest):
            raise ValueError("Judge identity changed or was not recorded")
        if evidence.get("status") != "completed":
            raise ValueError(f"Judge evaluation failed: {evidence.get('error')}")
        if name == "faithfulness":
            values = {"faithfulness": check_faithfulness_evidence(evidence, checksum)["value"]}
        elif name == "correctness":
            values = {"factual_correctness": check_correctness_evidence(evidence, checksum, sample, dataset)["value"]}
        else:
            checked = check_relevance_evidence(evidence, checksum, sample, dataset)
            values = {metric: checked[metric] for metric in EVIDENCE_METRICS[name]}
        return [{
                "name": metric,
                "metric": metric,
                "value": value,
                "minimum": minima[metric],
                "status": "measured" if minima[metric] is None else "passed" if value >= minima[metric] else "failed"}
                for metric, value in values.items()]
    except Exception as error:
        return [{
                "name": metric,
                "metric": metric,
                "status": "error",
                "error": f"{type(error).__name__}: {error}"} for metric in EVIDENCE_METRICS[name]]


def check_sample(path: Path, dataset: GoldenDataset, case: GoldenCase, model: str) -> dict:
    sample, _ = load_sample(path)
    bind_case(sample, dataset, case.id)
    metadata = sample.get("metadata", {})
    expected = {
            "golden_case_id": case.id,
            "golden_dataset_sha256": dataset.sha256,
            "policy_sha256": dataset.policy_sha256,
            "golden_category": case.category}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Benchmark sample lacks matching golden provenance")
    if sample["observation"]["request"]["model"] != model or not metadata.get("model_digest"):
        raise ValueError("Benchmark generation model/digest mismatch")
    if not isinstance(metadata.get("workspace_configuration"), dict):
        raise ValueError("Benchmark sample lacks workspace configuration")
    if "answer_request_seconds" in metadata:
        duration = metadata["answer_request_seconds"]
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Answer request duration must be a finite positive number")
    return sample


def acceptance_dimensions(sample: dict, case: GoldenCase) -> list[dict]:
    def sources() -> None:
        titles = {
                match[1]
                for context in sample["retrieved_contexts"]
                for match in re.finditer(r"^sourceDocument: (.+)$", context, re.MULTILINE)}
        if titles != {POLICY_DOCUMENT}:
            raise ValueError("Expected exactly one captured policy document")
        assertions.assert_document_sources({"sources": sample.get("response_sources")}, document_title=POLICY_DOCUMENT,
                fragments=case.source_fragments)

    dimensions = []
    for name, check in (("Reviewed answer rules", lambda: assertions.assert_golden_text(sample["response"], case=case)),
            ("Document sources", sources)):
        dimension = {"name": name, "status": "passed"}
        try:
            check()
        except AssertionError as error:
            dimension.update(status="failed", error=str(error))
        except Exception as error:
            dimension.update(status="error", error=f"{type(error).__name__}: {error}")
        dimensions.append(dimension)
    return dimensions


def evaluate_case(
        sample_path: Path, *, directory: Path, dataset: GoldenDataset, dataset_path: Path, policy_file: Path,
        case: GoldenCase, model: str, judge_model: str, judge_digest: str, faithfulness_model: str,
        faithfulness_digest: str, settings: Settings, minima: dict[str, float]) -> dict[str, Any]:
    sample = check_sample(sample_path, dataset, case, model)
    _, checksum = load_sample(sample_path)
    row: dict[str, Any] = {
            "case_id": case.id,
            "category": case.category,
            "model": model,
            "sample_sha256": checksum,
            "model_digest": sample["metadata"]["model_digest"],
            "workspace_configuration": sample["metadata"]["workspace_configuration"],
            "question": case.question,
            "reference": case.reference,
            "answer": sample["response"],
            "dimensions": acceptance_dimensions(sample, case),
            "judge_calls": 0,
            "judge_digests": {
            "faithfulness": faithfulness_digest,
            "correctness": judge_digest,
            "relevance": judge_digest},
            "evidence_sha256": {}}
    if "answer_request_seconds" in sample["metadata"]:
        row["answer_request_seconds"] = sample["metadata"]["answer_request_seconds"]
    if case.category == "missing_information":
        row["dimensions"].extend({
                "name": metric,
                "metric": metric,
                "status": "not_applicable",
                "reason": (
                "Abstention is checked against reviewed rules; semantic claim metrics are outside "
                "this benchmark's refusal scope")} for metric in sorted(minima))
        return row
    common = {
            "dataset_path": dataset_path,
            "policy_file": policy_file,
            "case_id": case.id,
            "settings": settings,
            "judge_model": judge_model}
    evaluators = ((
            "faithfulness",
            lambda: evaluate_sample_report(sample_path, settings=settings, judge_model=faithfulness_model)),
            ("correctness", lambda: evaluate_correctness_report(sample_path, **common)),
            ("relevance", lambda: evaluate_relevance_report(sample_path, **common)))
    for name, evaluate in evaluators:
        metrics = EVIDENCE_METRICS[name]
        try:
            evidence = evaluate()
            path = directory / f"{name}.json"
            write_sample(path, evidence)
            row["evidence_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
            row["judge_calls"] += sum(
                    len(evidence.get(k, [])) for k in ("judge_calls", "precision_calls", "recall_calls"))
            metric_judge = faithfulness_model if name == "faithfulness" else judge_model
            row["dimensions"].extend(
                    metric_dimensions(
                    name, evidence, checksum, sample, dataset, metric_judge, row["judge_digests"][name], minima))
        except Exception as error:
            row["dimensions"].extend({
                    "name": metric,
                    "metric": metric,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}"} for metric in metrics)
    return row
