"""Scenario data builders and deterministic test doubles."""

import json

from requests import Response

from test_support.paths import AUTOMATION_ROOT

DATA = AUTOMATION_ROOT / "test_data"


def response(answer: str, *, title: str = "policy.txt", source: str | None = None) -> Response:
    result = Response()
    result.status_code = 200
    result._content = json.dumps(
        {
            "type": "textResponse",
            "error": None,
            "close": True,
            "textResponse": answer,
            "sources": [
                {"title": title, "text": source or (DATA / "company-policy.txt").read_text()}
            ],
        }
    ).encode()
    return result
