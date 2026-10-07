"""Reviewed catalogs, references and parameter cases."""

from llm_testkit.datasets.golden import load_golden_dataset
from test_support.paths import AUTOMATION_ROOT

DATA_ROOT = AUTOMATION_ROOT / "test_data"


DATASET = load_golden_dataset(DATA_ROOT / "golden-policy.json", DATA_ROOT / "company-policy.txt")


REFERENCE_SATISFIES_ACCEPTANCE_CASE_CASES = DATASET.cases


def REFERENCE_SATISFIES_ACCEPTANCE_CASE_IDS(case):
    return case.id


INVALID_CATALOG_IS_REJECTED_MUTATION_MESSAGE_CASES = [
    ("duplicate", "duplicate golden case id"),
    ("checksum", "policy checksum mismatch"),
    ("category", "unknown category"),
    ("pattern", "invalid regex"),
    ("vacuous", "must not match empty text"),
    ("fragment", "absent from policy"),
    ("reference", "reference does not satisfy"),
    ("conflict", "reference violates forbidden"),
    ("schema", "integer schema_version"),
    ("empty", "must contain cases"),
]


MISSING_BENEFIT_CASE_ID_CASES = [
    "gym_missing",
    "bonus_missing",
    "parental_leave_missing",
]


BAD_EVIDENCE_IS_REJECTED_ANSWER_SOURCE_DOCUMENT_MESSAGE_CASES = [
    ("23 working days of paid leave.", None, "policy.txt", "advance notice"),
    (DATASET.cases[0].reference, "23 working days", "policy.txt", "12 calendar days"),
    (DATASET.cases[0].reference, None, "other.txt", "did not cite"),
    (
        "<think>23 working days and 12 calendar days before leave</think>No answer.",
        None,
        "policy.txt",
        "annual allowance",
    ),
]
