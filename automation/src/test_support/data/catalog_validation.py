"""Reviewed catalogs, references and parameter cases."""

from llm_testkit.datasets.adversarial import load_adversarial_cases
from llm_testkit.datasets.bias import load_bias_cases
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.datasets.prompts import load_prompt_catalog
from test_support.paths import AUTOMATION_ROOT

DATA = AUTOMATION_ROOT / "test_data"


DATASET = load_golden_dataset(DATA / "golden-policy.json", DATA / "company-policy.txt")


LOADERS = {
    "golden": (
        "golden-policy.json",
        "cases",
        lambda p: load_golden_dataset(p, DATA / "company-policy.txt"),
    ),
    "adversarial": (
        "adversarial-policy.json",
        "cases",
        lambda p: load_adversarial_cases(p, DATASET),
    ),
    "bias": ("bias-policy.json", "pairs", lambda p: load_bias_cases(p, DATASET)),
    "prompts": ("prompt-variants.json", "variants", load_prompt_catalog),
}


MALFORMED_CATALOG_CATALOG_CHANGE_CASES = [
    (name, change)
    for name in LOADERS
    for change in ("root", "row", "regex")
    if (name, change) != ("prompts", "regex")
]


DUPLICATE_JSON_FIELDS_CATALOG_CASES = LOADERS


NONSTRING_LOOKUP_CATALOG_FIELD_CASES = [
    ("bias", "golden_case_id"),
    ("adversarial", "category"),
    ("adversarial", "golden_case_id"),
    ("prompts", "baseline"),
]
