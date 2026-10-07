"""Scenario data builders and deterministic test doubles."""

import json

from requests import Response

from llm_testkit.datasets.golden import load_golden_dataset
from test_support.paths import AUTOMATION_ROOT

DATA_ROOT = AUTOMATION_ROOT / "test_data"


DATASET = load_golden_dataset(DATA_ROOT / "golden-policy.json", DATA_ROOT / "company-policy.txt")


def _response(answer: str, *, source: str | None = None, document: str = "policy.txt") -> Response:
    response = Response()
    response.status_code = 200
    response._content = json.dumps(
        {
            "type": "textResponse",
            "error": None,
            "close": True,
            "textResponse": answer,
            "sources": [
                {
                    "title": document,
                    "text": source or (DATA_ROOT / "company-policy.txt").read_text(),
                }
            ],
        }
    ).encode()
    return response
