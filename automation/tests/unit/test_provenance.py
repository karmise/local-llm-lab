from copy import deepcopy

import pytest

from llm_testkit.core.provenance import normalize_configuration
from test_support.assertions import errors as errors
from test_support.assertions import values as value_checks
from test_support.data.provenance import INVALID_CONFIGURATION_IS_NOT_COERCED_VALUE_CASES

pytestmark = pytest.mark.unit


def test_capture_suffix_is_incidental_and_configuration_is_not_mutated():
    marker = "[LLM_TESTKIT_CAPTURE:" + "a" * 32 + "]"
    original = {"openAiPrompt": "Policy\n" + marker, "options": {"topN": 4}}
    before = deepcopy(original)
    normalized = normalize_configuration(original)
    normalized["options"]["topN"] = 2
    value_checks.equal(normalized["openAiPrompt"], "Policy")
    value_checks.equal(original, before)
    value_checks.equal(
        normalize_configuration({"openAiPrompt": marker + "\nPolicy"})["openAiPrompt"],
        marker + "\nPolicy",
    )


@pytest.mark.parametrize("value", INVALID_CONFIGURATION_IS_NOT_COERCED_VALUE_CASES)
def test_invalid_configuration_is_not_coerced(value):
    errors.rejects(lambda: normalize_configuration(value), expected=ValueError)
