"""The local RAGAS judge: bounded call budget, deterministic options, strict structured output, retained evidence."""

import asyncio
import json
from unittest.mock import Mock

import pytest
from requests import HTTPError

from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.reporting.steps import title
from test_support.builders.identities import TEST_MODEL
from test_support.builders.ollama import chat_response
from test_support.builders.optional import load_ollama_judge

pytestmark = pytest.mark.unit

DETERMINISTIC_OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 2048}
CLAIMS = {"statements": ["Claim."]}


@pytest.fixture
def judge_class():
    return load_ollama_judge()


@pytest.fixture
def statements_model():
    pydantic = pytest.importorskip("pydantic")
    return pydantic.create_model("Statements", statements=(list[str], ...))


def expected_prompt(prompt: str, model) -> str:
    return f"{prompt}\nReturn JSON matching this schema:\n{json.dumps(model.model_json_schema())}"


@title("A judge defaults to two calls, a 300 s timeout and deterministic generation options")
def test_judge_defaults(judge_class):
    judge = judge_class(Mock(), TEST_MODEL)

    assert (judge.max_calls, judge.timeout, judge.calls) == (2, 300, [])
    assert judge.options == DETERMINISTIC_OPTIONS


@pytest.mark.parametrize("budget", [1, 4])
@title("A judge accepts a call budget on the bounds of one to four [{param_id}]")
def test_judge_accepts_budget_bounds(judge_class, budget):
    assert judge_class(Mock(), TEST_MODEL, max_calls=budget).max_calls == budget


@pytest.mark.parametrize("budget", [0, 5, 2.0, True], ids=["zero", "five", "float", "boolean"])
@title("A judge rejects a call budget that is not an integer from one to four [{param_id}]")
def test_judge_rejects_invalid_budget(judge_class, budget):
    with pytest.raises(ValueError, match="integer between one and four"):
        judge_class(Mock(), TEST_MODEL, max_calls=budget)


@title("A generation sends the prompt with its JSON schema and records the full exchange as evidence")
def test_generate_sends_schema_and_records_evidence(judge_class, statements_model):
    client = Mock()
    client.structured_chat.return_value = chat_response(CLAIMS)
    judge = judge_class(client, TEST_MODEL, timeout=30)
    prompt = expected_prompt("Extract claims", statements_model)

    parsed = judge.generate("Extract claims", statements_model)

    assert parsed.statements == ["Claim."]
    client.structured_chat.assert_called_once_with(
            model=TEST_MODEL, prompt=prompt, schema=statements_model.model_json_schema(), options=DETERMINISTIC_OPTIONS,
            timeout=30)
    assert judge.calls == [{
            "response_schema": "Statements",
            "prompt": prompt,
            "ollama_response": client.structured_chat.return_value.json(),
            "output": CLAIMS}]


@title("The judge request reaches Ollama as non-streaming chat with thinking disabled and the schema as format")
def test_generate_request_payload(judge_class, statements_model):
    http = Mock()
    http.request.return_value = chat_response(CLAIMS)
    judge = judge_class(OllamaClient(http), TEST_MODEL)

    judge.generate("Judge this", statements_model)

    http.request.assert_called_once_with(
            "POST", "/api/chat", timeout=300, json={
            "model": TEST_MODEL,
            "stream": False,
            "think": False,
            "keep_alive": "1m",
            "messages": [{
            "role": "user",
            "content": expected_prompt("Judge this", statements_model)}],
            "format": statements_model.model_json_schema(),
            "options": DETERMINISTIC_OPTIONS})


@pytest.mark.parametrize(("reply", "message"), [
        pytest.param(chat_response(CLAIMS, done_reason="length"), "incomplete or truncated", id="truncated"),
        pytest.param(chat_response(CLAIMS, done=False), "incomplete or truncated", id="unfinished"),
        pytest.param(chat_response(CLAIMS, model="other-model"), "unexpected model", id="other-model")])
@title("An incomplete or foreign judge reply is rejected once, without retry, and kept as evidence [{param_id}]")
def test_generate_rejects_unusable_reply_without_retry(judge_class, statements_model, reply, message):
    client = Mock()
    client.structured_chat.return_value = reply
    judge = judge_class(client, TEST_MODEL)

    with pytest.raises(ValueError, match=message):
        judge.generate("Extract claims", statements_model)

    client.structured_chat.assert_called_once()
    assert judge.calls[0]["ollama_response"] == reply.json()
    assert "output" not in judge.calls[0]


@title("Judge output is validated strictly, so a string is not coerced into an integer verdict")
def test_generate_validates_output_strictly(judge_class):
    pydantic = pytest.importorskip("pydantic")
    verdict_model = pydantic.create_model("Verdict", verdict=(int, ...))
    client = Mock()
    client.structured_chat.return_value = chat_response({"verdict": "1"})
    judge = judge_class(client, TEST_MODEL)

    with pytest.raises(pydantic.ValidationError):
        judge.generate("Verify the claim", verdict_model)


@title("An HTTP error from Ollama is raised after the attempted call is recorded")
def test_generate_raises_http_error(judge_class, statements_model):
    client = Mock()
    client.structured_chat.return_value = chat_response(CLAIMS, status_code=500)
    judge = judge_class(client, TEST_MODEL)

    with pytest.raises(HTTPError):
        judge.generate("Extract claims", statements_model)

    assert judge.calls == [{
            "response_schema": "Statements",
            "prompt": expected_prompt("Extract claims", statements_model)}]


@title("An exhausted budget is rejected before another transport call")
def test_generate_enforces_budget_before_calling_transport(judge_class, statements_model):
    client = Mock()
    client.structured_chat.return_value = chat_response(CLAIMS)
    judge = judge_class(client, TEST_MODEL, max_calls=1)
    judge.generate("Extract claims", statements_model)

    with pytest.raises(ValueError, match=r"budget exceeded \(maximum: 1\)"):
        judge.generate("Extract claims again", statements_model)

    assert client.structured_chat.call_count == 1
    assert len(judge.calls) == 1


@title("The asynchronous interface runs the same generation in a worker thread")
def test_agenerate_runs_generation(judge_class, statements_model):
    client = Mock()
    client.structured_chat.return_value = chat_response(CLAIMS)
    judge = judge_class(client, TEST_MODEL)

    parsed = asyncio.run(judge.agenerate("Extract claims", statements_model))

    assert parsed.statements == ["Claim."]
    assert judge.calls[0]["prompt"] == expected_prompt("Extract claims", statements_model)
