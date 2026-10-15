"""Base layer for API clients: a shared transport, a base path and default headers."""

from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from requests import Response

from llm_testkit.core.http_client import HttpClient


class BaseClient:
    """Sends requests for one API surface; subclasses only name operations and build payloads.

    Clients return raw responses and never check outcomes: acceptance criteria belong to `assertions`.
    """
    def __init__(self, http: HttpClient, *, base_path: str = "", headers: Mapping[str, str] | None = None) -> None:
        self._http = http
        self._base_path = base_path.rstrip("/")
        self._headers = dict(headers or {})

    def _request(self, method: str, path: str, **kwargs: Any) -> Response:
        if self._headers:
            kwargs["headers"] = self._headers | kwargs.get("headers", {})
        return self._http.request(method, f"{self._base_path}/{path.lstrip('/')}", **kwargs)

    @staticmethod
    def _segment(value: str) -> str:
        """`value` encoded as exactly one URL path segment, so a slug or folder name cannot change the route."""
        return quote(value, safe="")
