import json
import platform
from importlib.metadata import version

import pytest

from llm_testkit import assertions
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.iq


@title("Installation check confirms supported Python and pinned base test libraries")
def test_supported_runtime(request):
    request.node.user_properties.extend(
        [
            ("runtime_python", platform.python_version()),
            ("runtime_system", platform.platform()),
            (
                "runtime_libraries",
                json.dumps({name: version(name) for name in ("pytest", "requests")}),
            ),
        ]
    )
    assertions.assert_field_equals(
        {"python": platform.python_version_tuple()[:2]}, "python", ("3", "12")
    )
    for name, expected in {"pytest": "9.1.1", "requests": "2.34.2"}.items():
        assertions.assert_field_equals({"version": version(name)}, "version", expected)


@title("Installation check confirms both declared local generation models and their digests")
def test_declared_models_available(ollama_models, request):
    for model in ("qwen3.5:4b", "qwen2.5:7b"):
        assertions.assert_model_available(ollama_models, model)
    request.node.user_properties.append(("installed_models", json.dumps(ollama_models)))
