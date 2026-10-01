"""Shared HTTP transport for API clients."""

from typing import Any

import requests


class HttpClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self._timeout)
        return self._session.request(
            method=method,
            url=f"{self._base_url}/{path.lstrip('/')}",
            **kwargs,
        )

    def close(self) -> None:
        self._session.close()
