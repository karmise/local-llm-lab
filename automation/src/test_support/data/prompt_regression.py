"""Reviewed catalogs, references and parameter cases."""

from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.datasets.prompts import load_prompt_catalog
from test_support.data import common as case_data
from test_support.paths import AUTOMATION_ROOT

ROOT = AUTOMATION_ROOT


DATA = ROOT / "test_data"


CATALOG = load_prompt_catalog(DATA / "prompt-variants.json")


DATASET = load_golden_dataset(DATA / "golden-policy.json", DATA / "company-policy.txt")


COMPARISON_CHANGE_EXPECTED_CASES = [
    ("none", "passed"),
    ("failed", "regression"),
    ("baseline", "baseline_failed"),
    ("missing", "incomplete"),
    ("skipped", "incomplete"),
    ("error", "incomplete"),
    (case_data.MODEL_DIGEST, "incomplete"),
    ("prompt", "incomplete"),
    ("dataset", "incomplete"),
    ("duplicate", "incomplete"),
]


CATALOG_VALIDATION_CHANGE_CASES = ["duplicate", "baseline", "empty", "marker"]


CONFLICTING_PROPERTIES_WITHIN_ONE_TESTCASE_STALE_FIRST_CASES = [True, False]
