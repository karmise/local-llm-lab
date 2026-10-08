"""Scenario data builders and deterministic test doubles."""

import json

import pytest
from requests import Response

from llm_testkit.observation.evaluation_sample import build_sample
from test_support.data import common as case_data


def _sample() -> dict:
    capture_id = "a" * 32
    capture = {
            "schema_version": 1,
            "boundary": "ollama-sdk-chat",
            "request": {
            "model":
            case_data.TEST_MODEL,
            "stream":
            False,
            "messages": [{
            "role":
            "system",
            "content":
            f"[LLM_TESTKIT_CAPTURE:{capture_id}]\n"
            "[CONTEXT 0]:\nEmployees receive 23 working days.\n[END CONTEXT 0]"}, {
            "role": "user",
            "content": "How much leave?"}]}}
    return build_sample(
            capture, question="How much leave?", answer="Employees receive 23 working days.",
            reference="23 working days.", expected_model=case_data.TEST_MODEL, capture_id=capture_id)


def _judge_class():
    pytest.importorskip("ragas", reason="Install the evaluation extra for RAGAS adapter checks")
    from llm_testkit.evaluation.ollama_judge import OllamaJudge

    return OllamaJudge


def _response(output: dict, *, done_reason: str = "stop") -> Response:
    response = Response()
    response.status_code = 200
    response._content = json.dumps({
            "model": case_data.TEST_MODEL,
            "done": True,
            "done_reason": done_reason,
            "message": {
            "content": json.dumps(output)}}).encode()
    return response


def make_Statements_schema():
    from pydantic import BaseModel

    class Statements(BaseModel):
        statements: list[str]

    return Statements


def make_failed_score_stub():
    async def failed_score(sample: dict, judge: object) -> dict:
        raise ValueError("Judge generation truncated")

    return failed_score
