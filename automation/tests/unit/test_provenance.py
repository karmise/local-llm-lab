"""Configuration provenance: drop the per-run capture marker without changing experiment settings."""

from copy import deepcopy

import pytest

from llm_testkit.core.provenance import normalize_configuration
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit

MARKER = "[LLM_TESTKIT_CAPTURE:" + "a" * 32 + "]"


@title("A trailing capture marker is incidental and is removed")
def test_trailing_capture_marker_is_removed():
    assert normalize_configuration({
            "openAiPrompt": f"Policy\n{MARKER}",
            "topN": 4}) == {
            "openAiPrompt": "Policy",
            "topN": 4}


@pytest.mark.parametrize(
        "prompt", [
        pytest.param(f"{MARKER}\nPolicy", id="leading"),
        pytest.param(f"Policy {MARKER}", id="without-newline"),
        pytest.param(f"Policy\n{MARKER}\nMore", id="not-at-end"),
        pytest.param(f"Policy\n{MARKER}\n", id="before-final-newline"),
        pytest.param("Policy\n[LLM_TESTKIT_CAPTURE:" + "A" * 32 + "]", id="uppercase-id"),
        pytest.param("Policy\n[LLM_TESTKIT_CAPTURE:" + "a" * 31 + "]", id="short-id"),
        pytest.param("Policy\n[OTHER_CAPTURE:" + "a" * 32 + "]", id="other-marker")])
@title("Any prompt text other than a trailing capture marker is an experiment setting and is kept [{param_id}]")
def test_other_prompt_text_is_kept(prompt):
    assert normalize_configuration({"openAiPrompt": prompt})["openAiPrompt"] == prompt


@title("A configuration without a prompt is copied unchanged")
def test_configuration_without_prompt():
    assert normalize_configuration({"chatModel": "qwen", "topN": 4}) == {"chatModel": "qwen", "topN": 4}


@title("The original configuration is not mutated, even in nested settings")
def test_configuration_is_not_mutated():
    original = {"openAiPrompt": f"Policy\n{MARKER}", "options": {"topN": 4}}
    before = deepcopy(original)

    normalized = normalize_configuration(original)
    normalized["options"]["topN"] = 2

    assert original == before


@pytest.mark.parametrize("value", [pytest.param([], id="list"), pytest.param(None, id="none")])
@title("A configuration that is not an object is rejected, not coerced [{param_id}]")
def test_non_object_configuration_is_rejected(value):
    with pytest.raises(ValueError, match="^Workspace configuration must be an object$"):
        normalize_configuration(value)


@pytest.mark.parametrize("prompt", [pytest.param(1, id="number"), pytest.param(None, id="none")])
@title("A prompt that is not text is rejected, not coerced [{param_id}]")
def test_non_string_prompt_is_rejected(prompt):
    with pytest.raises(ValueError, match="^Workspace prompt must be a string$"):
        normalize_configuration({"openAiPrompt": prompt})
