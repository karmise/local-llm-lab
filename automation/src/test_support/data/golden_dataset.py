"""Reviewed catalogs, references and parameter cases."""

from llm_testkit.datasets.golden import load_golden_dataset
from test_support.data import common as case_data
from test_support.paths import AUTOMATION_ROOT

DATA_ROOT = AUTOMATION_ROOT / "test_data"

DATASET = load_golden_dataset(DATA_ROOT / "golden-policy.json", DATA_ROOT / "company-policy.txt")

# The loader validates every reference. Exercise the shared answer assertion once per category.
REFERENCE_ACCEPTANCE_CASES = tuple(
        case for case in DATASET.cases
        if case.id in {"paid_leave", "leave_approver", "hotel_receipt_condition", "gym_missing"})


def golden_case_id(case):
    return case.id


INVALID_GOLDEN_CATALOG_CASES = [("duplicate", "duplicate golden case id"), ("checksum", "policy checksum mismatch"),
        ("category", "unknown category"), ("pattern", "invalid regex"), ("vacuous", "must not match empty text"),
        ("fragment", "absent from policy"), ("reference", "reference does not satisfy"),
        ("conflict", "reference violates forbidden"), ("schema", "integer schema_version"),
        ("empty", "must contain cases")]

MISSING_BENEFIT_CASE_IDS = ["gym_missing", "bonus_missing", "parental_leave_missing"]

INVALID_ANSWER_EVIDENCE_CASES = [
        ("23 working days of paid leave.", None, case_data.POLICY_DOCUMENT_TITLE, "advance notice"),
        (DATASET.cases[0].reference, "23 working days", case_data.POLICY_DOCUMENT_TITLE, "12 calendar days"),
        (DATASET.cases[0].reference, None, "other.txt", "did not cite"),
        (
        "<think>23 working days and 12 calendar days before leave</think>No answer.", None,
        case_data.POLICY_DOCUMENT_TITLE, "annual allowance")]
