"""Scenario data builders and deterministic test doubles."""

import json

from requests import Response

from test_support.data import common as case_data


def _response(body: bytes) -> Response:
    response = Response()
    response.status_code = 200
    response._content = body
    return response


def _gym_response(answer: str) -> Response:
    payload = {
            "type":
            "textResponse",
            "error":
            None,
            "close":
            True,
            "textResponse":
            answer,
            "sources": [{
            "title": case_data.POLICY_DOCUMENT_TITLE,
            "text": "gym membership reimbursement policies are not covered"}]}
    return _response(json.dumps(payload).encode())


def make_answer_payload(answer, source_title, source_text):
    """Build input for test_rag_rejects_unsubstantiated_final_answers."""
    return {
            "type": "textResponse",
            "error": None,
            "close": True,
            "textResponse": answer,
            "sources": [{
            "title": source_title,
            "text": source_text}]}


def make_installed_model_entry():
    """Build input for test_model_selection_requires_a_digest."""
    return {"name": case_data.TEST_MODEL, "digest": ""}
