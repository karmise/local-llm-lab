"""Deterministic Ollama HTTP replies for judge adapters under test."""

import json
from typing import Any

from requests import Response

from test_support.builders.identities import TEST_MODEL


def chat_response(
        output: dict[str, Any], *, model: str = TEST_MODEL, done: bool = True, done_reason: str = "stop",
        status_code: int = 200) -> Response:
    """A structured-chat reply whose message content is the JSON-encoded judge output; completed by default."""
    response = Response()
    response.status_code = status_code
    response._content = json.dumps({
            "model": model,
            "done": done,
            "done_reason": done_reason,
            "message": {
            "content": json.dumps(output)}}).encode()
    return response


def model_catalog(*models: tuple[str, str], status_code: int = 200) -> Response:
    """An Ollama /api/tags reply listing (name, digest) pairs."""
    response = Response()
    response.status_code = status_code
    response._content = json.dumps({"models": [{"name": name, "digest": digest} for name, digest in models]}).encode()
    return response
