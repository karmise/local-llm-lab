"""Normalize incidental capture metadata without changing experiment settings."""

import re
from copy import deepcopy
from typing import Any

_CAPTURE_SUFFIX = re.compile(r"\n\[LLM_TESTKIT_CAPTURE:[a-f0-9]{32}\]\Z")


def normalize_configuration(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Workspace configuration must be an object")
    configuration = deepcopy(value)
    if "openAiPrompt" in configuration:
        prompt = configuration["openAiPrompt"]
        if not isinstance(prompt, str):
            raise ValueError("Workspace prompt must be a string")
        configuration["openAiPrompt"] = _CAPTURE_SUFFIX.sub("", prompt)
    return configuration
