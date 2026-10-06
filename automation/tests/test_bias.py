from pathlib import Path

import pytest

from llm_testkit import assertions
from llm_testkit.datasets.bias import BiasCase, load_bias_cases
from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.reporting.steps import title

pytestmark = [pytest.mark.rag, pytest.mark.bias]


@pytest.fixture
def bias_metadata(automation_root: Path, policy_file: Path, bias_case: BiasCase, record_property):
    root = automation_root / "test_data"
    dataset = load_golden_dataset(root / "golden-policy.json", policy_file)
    if bias_case not in load_bias_cases(root / "bias-policy.json", dataset):
        pytest.fail("Bias catalog changed after collection", pytrace=False)
    for name, value in {
        "bias_pair_id": bias_case.pair_id,
        "bias_variant_id": bias_case.variant_id,
        "bias_attribute": bias_case.attribute,
        "bias_catalog_sha256": bias_case.catalog_sha256,
        "bias_catalog_version": bias_case.catalog_version,
        "golden_case_id": bias_case.golden_case.id,
        "golden_dataset_sha256": dataset.sha256,
    }.items():
        record_property(name, value)


@title("Policy entitlements remain invariant for paired employee descriptors [{param_id}]")
def test_counterfactual_policy_answer(
    bias_case: BiasCase, bias_metadata, rag_chat, uploaded_policy_document
):
    response = rag_chat(bias_case.question, bias_case.golden_case.reference)
    assertions.assert_bias_answer(
        response, case=bias_case, document_title=uploaded_policy_document["title"]
    )
