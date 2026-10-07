"""Scenario data builders and deterministic test doubles."""

import json

from requests import Response

from llm_testkit.datasets.adversarial import load_adversarial_cases
from llm_testkit.datasets.golden import load_golden_dataset
from test_support.paths import AUTOMATION_ROOT

DATA = AUTOMATION_ROOT / "test_data"


DATASET = load_golden_dataset(DATA / "golden-policy.json", DATA / "company-policy.txt")


CASES = load_adversarial_cases(DATA / "adversarial-policy.json", DATASET)


def answer(case, text):
    response = Response()
    response.status_code = 200
    response._content = json.dumps(
        {
            "type": "textResponse",
            "textResponse": text,
            "close": True,
            "error": None,
            "sources": [{"title": "policy.txt", "text": (DATA / "company-policy.txt").read_text()}],
        }
    ).encode()
    return response
