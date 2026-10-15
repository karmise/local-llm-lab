"""Workspace operations of the AnythingLLM developer API."""

from collections.abc import Mapping
from typing import Any

from requests import Response

from api.base_client import BaseClient
from reporting.steps import step


class WorkspacesApi(BaseClient):
    """Workspace lifecycle, document indexing, vector search and chat."""
    def get_all(self) -> Response:
        return self._request("GET", "workspaces")

    def get(self, slug: str) -> Response:
        return self._request("GET", f"workspace/{self._segment(slug)}")

    @step("API: create workspace")
    def create(self, name: str, configuration: Mapping[str, Any] | None = None) -> Response:
        return self._request("POST", "workspace/new", json={**(configuration or {}), "name": name})

    @step("API: update workspace settings")
    def update(self, slug: str, configuration: Mapping[str, Any]) -> Response:
        return self._request("POST", f"workspace/{self._segment(slug)}/update", json=dict(configuration))

    @step("API: delete workspace")
    def delete(self, slug: str) -> Response:
        return self._request("DELETE", f"workspace/{self._segment(slug)}")

    @step("API: index workspace documents")
    def add_documents(self, slug: str, locations: list[str], *, timeout: float = 180) -> Response:
        payload = {"adds": locations, "deletes": []}
        return self._request(
                "POST", f"workspace/{self._segment(slug)}/update-embeddings", json=payload, timeout=timeout)

    @step("API: search workspace documents")
    def search(
            self, slug: str, query: str, *, top_n: int = 4, score_threshold: float = 0.25,
            timeout: float = 180) -> Response:
        payload = {"query": query, "topN": top_n, "scoreThreshold": score_threshold}
        return self._request("POST", f"workspace/{self._segment(slug)}/vector-search", json=payload, timeout=timeout)

    @step("API: generate workspace answer")
    def chat(self, slug: str, message: str, *, mode: str = "query", timeout: float = 300) -> Response:
        payload = {"message": message, "mode": mode}
        return self._request("POST", f"workspace/{self._segment(slug)}/chat", json=payload, timeout=timeout)
