"""Scenario data builders and deterministic test doubles."""

from llm_testkit.observation.evaluation_sample import build_sample

CAPTURE_ID = "a" * 32


def _capture() -> dict:
    return {
        "schema_version": 1,
        "boundary": "ollama-sdk-chat",
        "request": {
            "model": "qwen3.5:4b",
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Instructions\n[LLM_TESTKIT_CAPTURE:{CAPTURE_ID}]\nContext:\n"
                        "[CONTEXT 0]:\n23 working days\n[END CONTEXT 0]\n\n"
                        "[CONTEXT 1]:\n12 calendar days\n[END CONTEXT 1]\n\n"
                    ),
                },
                {"role": "user", "content": "Leave?"},
            ],
        },
    }


def _sample(capture: dict) -> dict:
    return build_sample(
        capture,
        question="Leave?",
        answer="23 working days",
        reference="Expected answer",
        expected_model="qwen3.5:4b",
        capture_id=CAPTURE_ID,
    )
