import pytest

import assertions
from builders.installation import DECLARED_MODELS, PINNED_LIBRARIES
from reporting.steps import title

pytestmark = pytest.mark.iq


@title("Installation check confirms supported Python and pinned base test libraries")
def test_supported_runtime(runtime_evidence):
    assertions.assert_supported_runtime(runtime_evidence, PINNED_LIBRARIES)


@title("Installation check confirms both declared local generation models and their digests")
def test_declared_models_available(installed_model_evidence):
    assertions.assert_models_available(installed_model_evidence, DECLARED_MODELS)
