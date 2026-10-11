"""Test data builders for reference-based factual correctness evidence."""

import hashlib
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from llm_testkit.evaluation.correctness import METRIC_CONFIGURATION
from test_support.builders.golden import GOLDEN_DATASET, PAID_LEAVE, TEST_DATA
from test_support.builders.identities import MODEL_DIGEST, TEST_MODEL

CONTROLS_FILE = TEST_DATA / "correctness-controls.json"


def labelled_claims(claims: Sequence[str], labels: Sequence[int]) -> list[dict[str, Any]]:
    """Judge verdicts for claims: label 1 means supported, 0 means not supported."""
    return [{
            "statement": claim,
            "verdict": label,
            "reason": "Test label"} for claim, label in zip(claims, labels, strict=True)]


def make_result(
        response_labels: Sequence[int] = (1, 1), reference_labels: Sequence[int] = (1, 1), value: float = 1.0, *,
        reference_claims: Sequence[str] | None = None) -> dict[str, Any]:
    """A factual-correctness result with labelled claims in both directions.

    Response labels mark answer claims supported by the reference (true or false positives).
    Reference labels mark reference claims covered by the answer (true positives or false negatives).
    """
    response_claims = [f"Response claim {i}" for i in range(len(response_labels))]
    if reference_claims is None:
        reference_claims = [f"Reference claim {i}" for i in range(len(reference_labels))]
    return {
            "value": value,
            "response_claims": response_claims,
            "reference_claims": list(reference_claims),
            "response_verdicts": labelled_claims(response_claims, response_labels),
            "reference_verdicts": labelled_claims(reference_claims, reference_labels)}


def judge_outputs(result: dict[str, Any]) -> list[dict[str, Any]]:
    """The four structured judge replies RAGAS requests for factual F1, in call order."""
    return [{
            "claims": result["response_claims"]}, {
            "statements": result["response_verdicts"]}, {
            "claims": result["reference_claims"]}, {
            "statements": result["reference_verdicts"]}]


def judge_calls(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Judge calls as OllamaJudge records them; independent copies of the result's claims."""
    return [{"output": output} for output in judge_outputs(deepcopy(result))]


def make_correctness_evidence(
        sample: dict[str, Any], *, sample_sha256: str = "sample",
        result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Completed correctness evidence bound to the sample, the golden case and its raw judge calls."""
    result = make_result() if result is None else result
    return {
            "schema_version": 1,
            "metric": "factual_correctness",
            "status": "completed",
            "sample_sha256": sample_sha256,
            "golden_dataset_sha256": GOLDEN_DATASET.sha256,
            "golden_case_id": PAID_LEAVE.id,
            "reference_sha256": hashlib.sha256(PAID_LEAVE.reference.encode()).hexdigest(),
            "response_origin": "application_sample",
            "question": PAID_LEAVE.question,
            "reference": PAID_LEAVE.reference,
            "response": sample["response"],
            "metric_configuration": deepcopy(METRIC_CONFIGURATION),
            "result": result,
            "judge_calls": judge_calls(result),
            "judge_model": TEST_MODEL,
            "judge_model_digest": MODEL_DIGEST}


def make_faithfulness_evidence(sample_sha256: str) -> dict[str, Any]:
    """Minimal completed faithfulness evidence for the same sample."""
    return {
            "schema_version": 1,
            "metric": "faithfulness",
            "status": "completed",
            "sample_sha256": sample_sha256,
            "result": {
            "value": 1.0,
            "statements": ["Claim"],
            "verdicts": [{
            "statement": "Claim",
            "verdict": 1}]},
            "judge_model": "judge",
            "judge_model_digest": MODEL_DIGEST,
            "judge_configuration": {},
            "ragas_version": "test",
            "created_at": "now"}
