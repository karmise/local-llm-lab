"""Collect installation evidence before runtime acceptance checks."""

import json
import platform
from importlib.metadata import version

import pytest

from test_support.data.installation import PINNED_LIBRARIES


@pytest.fixture
def runtime_evidence(request):
    runtime = {
        "python": platform.python_version(),
        "python_major_minor": platform.python_version_tuple()[:2],
        "system": platform.platform(),
        "libraries": {name: version(name) for name in PINNED_LIBRARIES},
    }
    request.node.user_properties.extend(
        [
            ("runtime_python", runtime["python"]),
            ("runtime_system", runtime["system"]),
            ("runtime_libraries", json.dumps(runtime["libraries"])),
        ]
    )
    return runtime


@pytest.fixture
def installed_model_evidence(ollama_models, request):
    request.node.user_properties.append(("installed_models", json.dumps(ollama_models)))
    return ollama_models
