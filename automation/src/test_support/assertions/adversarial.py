"""Domain expectations for adversarial unit scenarios."""

import hashlib

import pytest

from llm_testkit import assertions
from test_support.data.adversarial import DATASET as DATASET


def check_poisoned_copy_outcome(before, case, original, poisoned):
    if case.document_appendix:
        assert poisoned != original
        assert poisoned.read_text() == before.decode() + case.document_appendix
        assert hashlib.sha256(poisoned.read_bytes()).hexdigest() != DATASET.policy_sha256
        assertions.assert_attack_exposure([poisoned.read_text()], attack_text=case.document_appendix)
        with pytest.raises(AssertionError, match="not exposed"):
            assertions.assert_attack_exposure([original.read_text()], attack_text=case.document_appendix)
    else:
        assert poisoned == original
