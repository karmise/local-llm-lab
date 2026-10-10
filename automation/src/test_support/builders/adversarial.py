"""Test data builders for the reviewed adversarial attack catalog."""

import json
from typing import Any

from llm_testkit.datasets.adversarial import AdversarialCase, load_adversarial_cases
from test_support.builders.golden import GOLDEN_DATASET, TEST_DATA

ADVERSARIAL_FILE = TEST_DATA / "adversarial-policy.json"


def catalog_json() -> dict[str, Any]:
    """A fresh, editable copy of the reviewed attack catalog."""
    return json.loads(ADVERSARIAL_FILE.read_text())


def attack_cases() -> dict[str, AdversarialCase]:
    """The reviewed attacks by id, loaded on each call."""
    return {case.id: case for case in load_adversarial_cases(ADVERSARIAL_FILE, GOLDEN_DATASET)}
