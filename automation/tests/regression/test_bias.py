import pytest

import assertions
from fixtures.bias import bias_metadata as bias_metadata
from models.bias import BiasCase
from reporting.steps import title

pytestmark = [pytest.mark.rag, pytest.mark.bias]


@title("Policy entitlements remain invariant for paired employee descriptors [{param_id}]")
def test_counterfactual_policy_answer(bias_case: BiasCase, bias_metadata, rag_chat, uploaded_policy_document):
    response = rag_chat(bias_case.question, bias_case.golden_case.reference)
    assertions.assert_bias_answer(response, case=bias_case, document_title=uploaded_policy_document["title"])
