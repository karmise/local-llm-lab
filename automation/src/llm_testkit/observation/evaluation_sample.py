"""Build evaluation inputs from observed messages, never from response sources."""

import json
import re
from pathlib import Path
from typing import Any


def build_sample(
    capture: dict[str, Any],
    *,
    question: str,
    answer: str,
    reference: str,
    expected_model: str,
    capture_id: str,
) -> dict[str, Any]:
    if capture.get("schema_version") != 1 or capture.get("boundary") != "ollama-sdk-chat":
        raise ValueError("Unsupported observation boundary or schema")
    request = capture["request"]
    messages = request["messages"]
    if request.get("model") != expected_model or request.get("stream") is not False:
        raise ValueError("Expected the selected model and non-streaming chat")
    if len(messages) != 2 or [item.get("role") for item in messages] != ["system", "user"]:
        raise ValueError("Expected one system message and one user message, without history")
    if messages[1].get("content") != question:
        raise ValueError("Observed question does not match the scenario")
    system = messages[0]["content"]
    marker = f"[LLM_TESTKIT_CAPTURE:{capture_id}]"
    if system.count(marker) != 1:
        raise ValueError("Missing or ambiguous test capture marker")
    # This format belongs to AnythingLLM 1.16.2's Ollama constructPrompt.
    pattern = r"\[CONTEXT (\d+)\]:\n(.*?)\n\[END CONTEXT \1\]"
    matches = list(re.finditer(pattern, system, flags=re.DOTALL))
    remainder = re.sub(pattern, "", system, flags=re.DOTALL)
    if not matches or "[CONTEXT " in remainder or "[END CONTEXT " in remainder:
        raise ValueError("Missing, truncated or unsupported document context boundaries")
    if [int(match[1]) for match in matches] != list(range(len(matches))):
        raise ValueError("Context indices must be contiguous and ordered")
    contexts = [match[2] for match in matches]
    if any(not context.strip() for context in contexts):
        raise ValueError("Empty document context")
    return {
        "schema_version": 1,
        "user_input": question,
        "retrieved_contexts": contexts,
        "response": answer,
        "reference": reference,
        "observation": capture,
        "capture_id": capture_id,
        "context_parser": "anythingllm-1.16.2-ollama",
    }


def write_sample(path: Path, sample: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        path.chmod(0o600)
        json.dump(sample, output, ensure_ascii=False, indent=2)
        output.write("\n")
