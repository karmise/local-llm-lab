"""Reviewed catalogs, references and parameter cases."""

from llm_testkit.datasets.golden import load_golden_dataset
from test_support.paths import AUTOMATION_ROOT

ROOT = AUTOMATION_ROOT / "test_data"

DATASET = load_golden_dataset(ROOT / "golden-policy.json", ROOT / "company-policy.txt")

CASE = next(c for c in DATASET.cases if c.id == "paid_leave")

INVALID_RELEVANCE_CHANGE_CASES = ["binary", "reason", "score", "duplicate", "missing_fact", "count", "empty"]

EVIDENCE_BINDING_CHANGE_CASES = ["checksum", "dataset", "question", "raw_calls", "unfinished"]
