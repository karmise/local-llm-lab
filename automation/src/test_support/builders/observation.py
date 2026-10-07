"""Scenario data builders and deterministic test doubles."""

import pytest

from llm_testkit.observation.evaluation_sample import build_sample, write_sample
from test_support.data.observation import CAPTURE_ID as CAPTURE_ID


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


def prepare_sample_rejects_mismatched_or_ambiguous_observations_case(change, request, system):
    if change == "model":
        request["model"] = "other-model"
    elif change == "question":
        request["messages"][1]["content"] = "Other question"
    elif change == "history":
        request["messages"].append({"role": "assistant", "content": "History"})
    elif change == "marker":
        system["content"] = system["content"].replace(CAPTURE_ID, "b" * 32)
    elif change == "truncated":
        system["content"] = system["content"].replace("[END CONTEXT 1]", "")
    elif change == "indices":
        system["content"] = system["content"].replace("CONTEXT 1", "CONTEXT 2")
    else:
        system["content"] = system["content"].replace("23 working days", "")


def prepare_sdk_hook_preserves_request_return_values_streams_and_errors_case(node):
    if not node:
        pytest.skip("Node.js is required to verify the optional application preload")


def make_publish_stub(path):
    def publish(index):
        try:
            write_sample(path, {"writer": index, "payload": "content" * 1000})
            return index
        except FileExistsError:
            return None

    return publish


def make_fail_stub():
    def fail(*args):
        raise OSError("publication unavailable")

    return fail


def check_sdk_hook_preserves_request_return_values_streams_and_errors_step_7(actual):
    for field in ("sameRequest", "sameReturn", "sameStream", "sameError"):
        assert actual[field] is True
