"""Native Ollama API operations used by the judge and model checks."""

from typing import Any

from requests import Response

from llm_testkit.clients.base_client import BaseClient


class OllamaClient(BaseClient):
    """Installed models and one structured, non-streaming chat request."""
    def list_models(self) -> Response:
        return self._request("GET", "/api/tags")

    def structured_chat(
            self, *, model: str, prompt: str, schema: dict[str, Any], options: dict[str, Any],
            timeout: float = 300) -> Response:
        message = {"role": "user", "content": prompt}
        payload = {
                "model": model,
                "stream": False,
                "think": False,
                "keep_alive": "1m",
                "messages": [message],
                "format": schema,
                "options": options}
        return self._request("POST", "/api/chat", timeout=timeout, json=payload)
