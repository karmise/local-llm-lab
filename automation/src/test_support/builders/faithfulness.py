"""Test data builders for faithfulness evaluation."""

from typing import Any

from llm_testkit.observation.evaluation_sample import build_sample
from test_support.data.common import TEST_MODEL

QUESTION = "How much leave?"
ANSWER = "Employees receive 23 working days."
CONTEXT = "Employees receive 23 working days."


def make_sample(*, context: str = CONTEXT) -> dict[str, Any]:
    """A captured sample with one retrieved context; the answer repeats the context."""
    capture_id = "a" * 32
    system_prompt = f"[LLM_TESTKIT_CAPTURE:{capture_id}]\n[CONTEXT 0]:\n{context}\n[END CONTEXT 0]"
    capture = {
            "schema_version": 1,
            "boundary": "ollama-sdk-chat",
            "request": {
            "model": TEST_MODEL,
            "stream": False,
            "messages": [{
            "role": "system",
            "content": system_prompt}, {
            "role": "user",
            "content": QUESTION}]}}
    return build_sample(
            capture, question=QUESTION, answer=ANSWER, reference="23 working days.", expected_model=TEST_MODEL,
            capture_id=capture_id)


def make_result(labels: tuple[int, ...] = (1, 0), value: float | None = None) -> dict[str, Any]:
    """A faithfulness result with one statement per label; the value defaults to the supported share."""
    statements = [f"Claim {i}." for i in range(len(labels))]
    verdicts = [{"statement": s, "verdict": label} for s, label in zip(statements, labels, strict=True)]
    return {
            "value": sum(labels) / len(labels) if value is None else value,
            "statements": statements,
            "verdicts": verdicts}
