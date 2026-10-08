"""Reviewed catalogs, references and parameter cases."""

from llm_testkit.datasets.bias import load_bias_cases
from llm_testkit.datasets.golden import load_golden_dataset
from test_support.data import common as case_data
from test_support.paths import AUTOMATION_ROOT

DATA = AUTOMATION_ROOT / "test_data"

DATASET = load_golden_dataset(DATA / "golden-policy.json", DATA / "company-policy.txt")

CASES = load_bias_cases(DATA / "bias-policy.json", DATASET)

ANSWERS_CASE_CASES = CASES

ANSWERS_CASE_IDS = [f"{c.pair_id}-{c.variant_id}" for c in CASES]

CATALOG_CHANGE_CASES = ["dataset", "duplicate", "descriptor", "question", "variants", "regex"]

COMPARISON_CHANGE_STATUS_OUTCOME_CASES = [("none", "passed", "passed"), ("one_failure", "failed", "asymmetry"),
        ("two_failures", "failed", "shared_failure"), ("missing", "incomplete", "incomplete"),
        ("error", "incomplete", "incomplete"), (case_data.MODEL_DIGEST, "incomplete", "incomplete"),
        ("duplicate", "incomplete", "incomplete")]
