"""Deterministic Ollama HTTP replies for judge adapters under test."""

import json
from typing import Any

from requests import Response

from test_support.data.common import TEST_MODEL


def chat_response(output: dict[str, Any], *, model: str = TEST_MODEL, done_reason: str = "stop") -> Response:
    """A completed structured-chat reply whose message content is the JSON-encoded judge output."""
    response = Response()
    response.status_code = 200
    response._content = json.dumps({
            "model": model,
            "done": True,
            "done_reason": done_reason,
            "message": {
            "content": json.dumps(output)}}).encode()
    return response


def model_catalog(*models: tuple[str, str]) -> Response:
    """An Ollama /api/tags reply listing (name, digest) pairs."""
    response = Response()
    response.status_code = 200
    response._content = json.dumps({"models": [{"name": name, "digest": digest} for name, digest in models]}).encode()
    return response
