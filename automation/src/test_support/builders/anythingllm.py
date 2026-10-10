"""Deterministic AnythingLLM chat replies for answer and source checks."""

import json

from requests import Response

from test_support.builders.golden import POLICY_FILE
from test_support.data.common import POLICY_DOCUMENT_TITLE


def chat_reply(answer: str, *, source: str | None = None, document: str = POLICY_DOCUMENT_TITLE) -> Response:
    """A completed text reply citing one document; the cited text defaults to the whole policy."""
    payload = {
            "type": "textResponse",
            "error": None,
            "close": True,
            "textResponse": answer,
            "sources": [{
            "title": document,
            "text": POLICY_FILE.read_text() if source is None else source}]}
    response = Response()
    response.status_code = 200
    response._content = json.dumps(payload).encode()
    return response
