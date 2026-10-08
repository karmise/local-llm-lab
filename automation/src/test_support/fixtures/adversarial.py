"""Scoped scenario fixtures."""

from pathlib import Path

import pytest

from llm_testkit.datasets.adversarial import AdversarialCase, load_adversarial_cases, materialize_policy
from llm_testkit.datasets.golden import load_golden_dataset


@pytest.fixture
def adversarial_metadata(automation_root: Path, adversarial_case: AdversarialCase, record_property):
    # Bind expectations to the original policy, never the poisoned test copy.
    data = automation_root / "test_data"
    dataset = load_golden_dataset(data / "golden-policy.json", data / "company-policy.txt")
    cases = load_adversarial_cases(data / "adversarial-policy.json", dataset)
    if adversarial_case not in cases:
        pytest.fail("Adversarial inputs changed after collection", pytrace=False)
    for name, value in {"adversarial_case_id": adversarial_case.id, "adversarial_category": adversarial_case.category,
            "adversarial_catalog_sha256": adversarial_case.catalog_sha256, "adversarial_catalog_version":
            adversarial_case.catalog_version, "golden_case_id": adversarial_case.golden_case.id,
            "golden_dataset_sha256": dataset.sha256, "adversarial_base_policy_sha256": dataset.policy_sha256}.items():
        record_property(name, value)


@pytest.fixture
def policy_file(automation_root: Path, tmp_path: Path, adversarial_case: AdversarialCase) -> Path:
    return materialize_policy(
            automation_root / "test_data/company-policy.txt", adversarial_case, tmp_path / "company-policy.txt")
