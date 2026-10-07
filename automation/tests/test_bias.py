import pytest

from llm_testkit import assertions
from llm_testkit.datasets.bias import BiasCase
from llm_testkit.reporting.steps import title
from test_support.fixtures.bias import bias_metadata as bias_metadata

pytestmark = [pytest.mark.rag, pytest.mark.bias]


@title("Policy entitlements remain invariant for paired employee descriptors [{param_id}]")
def test_counterfactual_policy_answer(bias_case: BiasCase, bias_metadata, rag_chat,
                                      uploaded_policy_document):  # fmt: skip
    response = rag_chat(bias_case.question, bias_case.golden_case.reference)
    assertions.assert_bias_answer(
        response, case=bias_case, document_title=uploaded_policy_document["title"]
    )
