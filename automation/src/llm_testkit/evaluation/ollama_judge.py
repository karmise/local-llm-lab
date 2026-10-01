"""RAGAS structured generation using the native Ollama API, without retries."""

import asyncio
import json
from typing import Any, TypeVar

from pydantic import BaseModel
from ragas.llms.base import InstructorBaseRagasLLM

from llm_testkit.clients.ollama_client import OllamaClient

T = TypeVar("T", bound=BaseModel)


class OllamaJudge(InstructorBaseRagasLLM):
    def __init__(self, client: OllamaClient, model: str, timeout: float = 300) -> None:
        self.client = client
        self.model = model
        self.timeout = timeout
        self.options = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 2048}
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, response_model: type[T]) -> T:
        if len(self.calls) >= 2:
            raise ValueError("Faithfulness judge call budget exceeded (maximum: 2)")
        schema = response_model.model_json_schema()
        full_prompt = f"{prompt}\nReturn JSON matching this schema:\n{json.dumps(schema)}"
        call: dict[str, Any] = {"response_schema": response_model.__name__, "prompt": full_prompt}
        self.calls.append(call)
        response = self.client.structured_chat(
            model=self.model, prompt=full_prompt, schema=schema,
            options=self.options, timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        call["ollama_response"] = payload
        if payload.get("done") is not True or payload.get("done_reason") != "stop":
            raise ValueError("Judge generation is incomplete or truncated")
        if payload.get("model") != self.model:
            raise ValueError("Judge returned an unexpected model")
        parsed = response_model.model_validate_json(payload["message"]["content"], strict=True)
        call["output"] = parsed.model_dump()
        return parsed

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        return await asyncio.to_thread(self.generate, prompt, response_model)
