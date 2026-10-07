"""Scenario data builders and deterministic test doubles."""

import json

from requests import Response


def _response(body: bytes) -> Response:
    response = Response()
    response.status_code = 200
    response._content = body
    return response


def _gym_response(answer: str) -> Response:
    payload = {
        "type": "textResponse",
        "error": None,
        "close": True,
        "textResponse": answer,
        "sources": [
            {"title": "policy.txt", "text": "gym membership reimbursement policies are not covered"}
        ],
    }
    return _response(json.dumps(payload).encode())
