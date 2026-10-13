import pytest

from llm_testkit import assertions
from llm_testkit.reporting.steps import title
from test_support.builders.installation import DECLARED_MODELS, PINNED_LIBRARIES

pytestmark = pytest.mark.iq


@title("Installation check confirms supported Python and pinned base test libraries")
def test_supported_runtime(runtime_evidence):
    assertions.assert_supported_runtime(runtime_evidence, PINNED_LIBRARIES)


@title("Installation check confirms both declared local generation models and their digests")
def test_declared_models_available(installed_model_evidence):
    assertions.assert_models_available(installed_model_evidence, DECLARED_MODELS)
