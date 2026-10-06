from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.datasets.adversarial import (
    AdversarialCase,
    load_adversarial_cases,
    materialize_policy,
)
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.evaluation.faithfulness import load_sample
from llm_testkit.reporting.steps import title

pytestmark = [pytest.mark.rag, pytest.mark.adversarial]


@pytest.fixture
def adversarial_metadata(automation_root: Path, adversarial_case: AdversarialCase, record_property):
    # Bind expectations to the original policy, never the poisoned test copy.
    data = automation_root / "test_data"
    dataset = load_golden_dataset(data / "golden-policy.json", data / "company-policy.txt")
    cases = load_adversarial_cases(data / "adversarial-policy.json", dataset)
    if adversarial_case not in cases:
        pytest.fail("Adversarial inputs changed after collection", pytrace=False)
    for name, value in {
        "adversarial_case_id": adversarial_case.id,
        "adversarial_category": adversarial_case.category,
        "adversarial_catalog_sha256": adversarial_case.catalog_sha256,
        "adversarial_catalog_version": adversarial_case.catalog_version,
        "golden_case_id": adversarial_case.golden_case.id,
        "golden_dataset_sha256": dataset.sha256,
        "adversarial_base_policy_sha256": dataset.policy_sha256,
    }.items():
        record_property(name, value)


@pytest.fixture
def policy_file(automation_root: Path, tmp_path: Path, adversarial_case: AdversarialCase) -> Path:
    return materialize_policy(
        automation_root / "test_data/company-policy.txt",
        adversarial_case,
        tmp_path / "company-policy.txt",
    )


@title("Policy answer resists curated adversarial input [{param_id}]")
def test_policy_resists_adversarial_input(
    adversarial_case: AdversarialCase,
    adversarial_metadata: None,
    rag_chat: Callable[[str, str], Response],
    uploaded_policy_document: dict[str, Any],
    capture_id: str | None,
    automation_root: Path,
) -> None:
    response = rag_chat(adversarial_case.question, adversarial_case.golden_case.reference)
    if adversarial_case.document_appendix:
        sample, _ = load_sample(automation_root / f"reports/rag-samples/{capture_id}.json")
        assertions.assert_attack_exposure(
            sample["retrieved_contexts"], attack_text=adversarial_case.document_appendix
        )
    assertions.assert_adversarial_answer(
        response, case=adversarial_case, document_title=uploaded_policy_document["title"]
    )
