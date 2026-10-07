from copy import deepcopy

import pytest

from llm_testkit.core.provenance import normalize_configuration
from test_support.data.provenance import INVALID_CONFIGURATION_IS_NOT_COERCED_VALUE_CASES

pytestmark = pytest.mark.unit


def test_capture_suffix_is_incidental_and_configuration_is_not_mutated():
    marker = "[LLM_TESTKIT_CAPTURE:" + "a" * 32 + "]"
    original = {"openAiPrompt": "Policy\n" + marker, "options": {"topN": 4}}
    before = deepcopy(original)
    normalized = normalize_configuration(original)
    normalized["options"]["topN"] = 2
    assert normalized["openAiPrompt"] == "Policy"
    assert original == before
    assert (
        normalize_configuration({"openAiPrompt": marker + "\nPolicy"})["openAiPrompt"]
        == marker + "\nPolicy"
    )


@pytest.mark.parametrize("value", INVALID_CONFIGURATION_IS_NOT_COERCED_VALUE_CASES)
def test_invalid_configuration_is_not_coerced(value):
    with pytest.raises(ValueError):
        normalize_configuration(value)
