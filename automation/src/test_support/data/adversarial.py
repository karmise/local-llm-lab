"""Reviewed catalogs, references and parameter cases."""

from llm_testkit.datasets.adversarial import load_adversarial_cases
from llm_testkit.datasets.golden import load_golden_dataset
from test_support.paths import AUTOMATION_ROOT

DATA = AUTOMATION_ROOT / "test_data"

DATASET = load_golden_dataset(DATA / "golden-policy.json", DATA / "company-policy.txt")

CASES = load_adversarial_cases(DATA / "adversarial-policy.json", DATASET)

ASSERTIONS_CASE_CASES = CASES

ASSERTIONS_CASE_IDS = [c.id for c in CASES]

POISONED_COPY_CASE_CASES = CASES

POISONED_COPY_CASE_IDS = [c.id for c in CASES]

CATALOG_CHANGE_CASES = ["hash", "id", "category", "base", "question", "appendix", "regex"]
