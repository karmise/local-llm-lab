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
