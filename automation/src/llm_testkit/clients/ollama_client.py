"""Native Ollama model metadata operations."""

from requests import Response
from typing import Any

from llm_testkit.core.http_client import HttpClient


class OllamaClient:
    def __init__(self, http_client: HttpClient) -> None:
        self._http = http_client

    def list_models(self) -> Response:
        return self._http.request("GET", "/api/tags")

    def structured_chat(
        self, *, model: str, prompt: str, schema: dict[str, Any],
        options: dict[str, Any], timeout: float = 300,
    ) -> Response:
        return self._http.request(
            "POST", "/api/chat", timeout=timeout,
            json={
                "model": model, "stream": False, "think": False, "keep_alive": "1m",
                "messages": [{"role": "user", "content": prompt}],
                "format": schema, "options": options,
            },
        )
