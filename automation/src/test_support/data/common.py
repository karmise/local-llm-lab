"""Shared synthetic identities and copies of mutable unit inputs."""

from copy import deepcopy
from typing import TypeVar

T = TypeVar("T")

TEST_MODEL = "test-model"
MODEL_DIGEST = "digest"
SAMPLE_FILE_NAME = "sample.json"
POLICY_DOCUMENT_TITLE = "policy.txt"
EMPTY_OBJECT = {}


def fresh(case: T) -> T:
    """Every test receives its own nested payload; parameter catalogs stay unchanged."""
    return deepcopy(case)
