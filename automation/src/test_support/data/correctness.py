"""Reviewed catalogs, references and parameter cases."""

from llm_testkit.datasets.golden import load_golden_dataset
from test_support.paths import AUTOMATION_ROOT

ROOT = AUTOMATION_ROOT / "test_data"


DATASET = load_golden_dataset(ROOT / "golden-policy.json", ROOT / "company-policy.txt")


CASE = next(case for case in DATASET.cases if case.id == "paid_leave")


REAL_RAGAS_CORRECTNESS_WITH_MOCKED_JUDGE_RV_GV_EXPECTED_CASES = [
    ((1, 1), (1, 1), 1.0),
    ((1,), (1, 0), 0.67),
    ((0, 0), (0, 0), 0.0),
    ((1, 1, 0), (1, 1), 0.8),
]


REAL_RAGAS_CORRECTNESS_WITH_MOCKED_JUDGE_RV_GV_EXPECTED_IDS = [
    "correct",
    "incomplete",
    "contradicted",
    "extra-claim",
]


INVALID_CLAIM_EVIDENCE_IS_REJECTED_CHANGE_CASES = [
    "missing",
    "duplicate",
    "invalid-verdict",
    "score",
    "counts",
    "empty",
    "reason",
]


UNRELATED_OR_SYNTHETIC_EVIDENCE_IS_REJECTED_CHANGE_CASES = [
    "sample",
    "dataset",
    "reference",
    "response",
    "config",
    "control",
    "unfinished",
]


# Input for test_unrelated_or_synthetic_evidence_is_rejected
EVIDENCE_MUTATION_FIELDS = {
    "sample": "sample_sha256",
    "dataset": "golden_dataset_sha256",
    "reference": "reference",
    "response": "response",
    "config": "metric_configuration",
    "control": "response_origin",
    "unfinished": "status",
}


# Input for test_sample_must_use_current_golden_reference
STALE_DATASET_METADATA = {"golden_dataset_sha256": "stale"}


INCOMPLETE_REFERENCE_CLAIMS = [
    "Each employee receives 23 working days of paid leave.",
    "A request needs 12 calendar days before leave starts.",
]
MODIFIED_EVIDENCE_VALUE = "changed"
EDITED_RAW_CLAIMS = ["Edited"]
