"""Scoped scenario fixtures."""

from pathlib import Path

import pytest

from llm_testkit.datasets.bias import BiasCase, load_bias_cases
from llm_testkit.datasets.golden import load_golden_dataset


@pytest.fixture
def bias_metadata(automation_root: Path, policy_file: Path, bias_case: BiasCase, record_property):
    root = automation_root / "test_data"
    dataset = load_golden_dataset(root / "golden-policy.json", policy_file)
    if bias_case not in load_bias_cases(root / "bias-policy.json", dataset):
        pytest.fail("Bias catalog changed after collection", pytrace=False)
    for name, value in {"bias_pair_id": bias_case.pair_id, "bias_variant_id": bias_case.variant_id, "bias_attribute":
            bias_case.attribute, "bias_catalog_sha256": bias_case.catalog_sha256, "bias_catalog_version":
            bias_case.catalog_version, "golden_case_id": bias_case.golden_case.id, "golden_dataset_sha256":
            dataset.sha256}.items():
        record_property(name, value)
