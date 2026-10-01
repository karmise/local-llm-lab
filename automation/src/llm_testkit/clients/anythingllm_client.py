"""AnythingLLM API operations, without test assertions."""

from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from requests import Response

from llm_testkit.core.http_client import HttpClient


class AnythingLLMClient:
    def __init__(self, http_client: HttpClient, api_key: str | None = None) -> None:
        self._http = http_client
        self._api_key = api_key

    def health(self) -> Response:
        return self._http.request("GET", "/api/ping")

    def _developer_request(self, method: str, path: str, **kwargs: Any) -> Response:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        return self._http.request(method, f"/api/v1/{path}", headers=headers, **kwargs)

    def verify_authentication(self) -> Response:
        return self._developer_request("GET", "auth")

    def get_workspace(self, slug: str) -> Response:
        return self._developer_request("GET", f"workspace/{quote(slug, safe='')}")

    def create_workspace(
        self, name: str, configuration: Mapping[str, Any] | None = None
    ) -> Response:
        payload = {**(configuration or {}), "name": name}
        return self._developer_request("POST", "workspace/new", json=payload)

    def delete_workspace(self, slug: str) -> Response:
        return self._developer_request("DELETE", f"workspace/{quote(slug, safe='')}")

    def create_document_folder(self, name: str) -> Response:
        return self._developer_request("POST", "document/create-folder", json={"name": name})

    def delete_document_folder(self, name: str, *, timeout: float = 180) -> Response:
        return self._developer_request(
            "DELETE", "document/remove-folder", json={"name": name}, timeout=timeout
        )

    def get_document_folder(self, name: str) -> Response:
        return self._developer_request("GET", f"documents/folder/{quote(name, safe='')}")

    def upload_document(
        self, path: Path, folder: str, *, filename: str | None = None, timeout: float = 180
    ) -> Response:
        with path.open("rb") as document:
            return self._developer_request(
                "POST",
                f"document/upload/{quote(folder, safe='')}",
                files={"file": (filename or path.name, document, "text/plain")},
                timeout=timeout,
            )

    def add_workspace_documents(
        self, slug: str, locations: list[str], *, timeout: float = 180
    ) -> Response:
        return self._developer_request(
            "POST", f"workspace/{quote(slug, safe='')}/update-embeddings",
            json={"adds": locations, "deletes": []}, timeout=timeout,
        )

    def search_workspace(
        self, slug: str, query: str, *, top_n: int = 4,
        score_threshold: float = 0.25, timeout: float = 180,
    ) -> Response:
        return self._developer_request(
            "POST", f"workspace/{quote(slug, safe='')}/vector-search",
            json={"query": query, "topN": top_n, "scoreThreshold": score_threshold},
            timeout=timeout,
        )

    def chat(
        self, slug: str, message: str, *, mode: str = "query", timeout: float = 300
    ) -> Response:
        return self._developer_request(
            "POST", f"workspace/{quote(slug, safe='')}/chat",
            json={"message": message, "mode": mode}, timeout=timeout,
        )
