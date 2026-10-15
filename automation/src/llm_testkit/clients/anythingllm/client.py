"""Entry point to the AnythingLLM API."""

from requests import Response

from llm_testkit.clients.anythingllm.documents import DocumentsApi
from llm_testkit.clients.anythingllm.workspaces import WorkspacesApi
from llm_testkit.clients.base_client import BaseClient
from llm_testkit.core.http_client import HttpClient

DEVELOPER_API = "/api/v1"


class AnythingLLMClient(BaseClient):
    """System checks, plus `workspaces` and `documents` operations sharing one transport and API key."""
    def __init__(self, http: HttpClient, api_key: str | None = None) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        super().__init__(http, base_path=DEVELOPER_API, headers=headers)
        self.workspaces = WorkspacesApi(http, base_path=DEVELOPER_API, headers=headers)
        self.documents = DocumentsApi(http, base_path=DEVELOPER_API, headers=headers)

    def health(self) -> Response:
        # The ping endpoint is outside the developer API and needs no key.
        return self._http.request("GET", "/api/ping")

    def verify_authentication(self) -> Response:
        return self._request("GET", "auth")
