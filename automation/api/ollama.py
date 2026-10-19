"""Native Ollama API operations used by the judge and model checks."""

from typing import Any

from requests import Response

from api.base_client import BaseClient


class OllamaClient(BaseClient):
    """Installed models and single non-streaming chat requests, free-form or structured."""
    def list_models(self) -> Response:
        return self._request("GET", "/api/tags")

    def chat(self, *, model: str, prompt: str, options: dict[str, Any], timeout: float = 300) -> Response:
        payload = {
                "model": model,
                "stream": False,
                "keep_alive": "1m",
                "messages": [{
                "role": "user",
                "content": prompt}],
                "options": options}
        return self._request("POST", "/api/chat", timeout=timeout, json=payload)

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
