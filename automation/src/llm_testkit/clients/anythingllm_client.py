"""AnythingLLM API operations, without test assertions."""

from urllib.parse import quote

from requests import Response

from llm_testkit.core.http_client import HttpClient


class AnythingLLMClient:
    def __init__(self, http_client: HttpClient, api_key: str | None = None) -> None:
        self._http = http_client
        self._api_key = api_key

    def health(self) -> Response:
        return self._http.request("GET", "/api/ping")

    def _developer_request(self, method: str, path: str) -> Response:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        return self._http.request(method, f"/api/v1/{path}", headers=headers)

    def verify_authentication(self) -> Response:
        return self._developer_request("GET", "auth")

    def get_workspace(self, slug: str) -> Response:
        return self._developer_request("GET", f"workspace/{quote(slug, safe='')}")
