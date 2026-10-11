"""Test data builders for reference-based context precision and recall."""

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from test_support.builders.golden import GOLDEN_DATASET, PAID_LEAVE

# Reference claims that satisfy the paid-leave case's labelled facts.
ALLOWANCE_CLAIM = "Employees receive 23 working days of paid leave."
NOTICE_CLAIM = "Request at least 12 calendar days before leave."


def precision_verdicts(labels: Sequence[int]) -> list[dict[str, Any]]:
    """One judge verdict per retrieved context, in retrieval order: 1 means the context is useful."""
    return [{"verdict": label, "reason": "Labelled context"} for label in labels]


def recall_classifications(
        attributed: Sequence[int] = (1, 0), statements: Sequence[str] = (ALLOWANCE_CLAIM, NOTICE_CLAIM)) -> list[dict]:
    """Reference claims classified as attributed (1) or not (0) to the retrieved contexts."""
    return [{
            "statement": statement,
            "attributed": label,
            "reason": "Labelled claim"} for statement, label in zip(statements, attributed, strict=True)]


def make_result(
        precision_labels: Sequence[int] = (1, 0, 1), context_precision: float = 5 / 6,
        attributed: Sequence[int] = (1, 0), context_recall: float = 0.5) -> dict[str, Any]:
    """A relevance result; the default scores are the average precision of (1, 0, 1) and recall of (1, 0)."""
    return {
            "context_precision": context_precision,
            "context_recall": context_recall,
            "precision_verdicts": precision_verdicts(precision_labels),
            "recall_classifications": recall_classifications(attributed)}


def make_relevance_evidence(*, sample_sha256: str = "sample", result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Completed relevance evidence with raw judge calls that reproduce the result."""
    result = make_result() if result is None else result
    return {
            "schema_version": 1,
            "metric": "context_relevance",
            "status": "completed",
            "sample_sha256": sample_sha256,
            "golden_dataset_sha256": GOLDEN_DATASET.sha256,
            "golden_case_id": PAID_LEAVE.id,
            "question": PAID_LEAVE.question,
            "reference": PAID_LEAVE.reference,
            "result": result,
            "precision_calls": [{
            "output": verdict} for verdict in deepcopy(result["precision_verdicts"])],
            "recall_calls": [{
            "output": {
            "classifications": deepcopy(result["recall_classifications"])}}]}
