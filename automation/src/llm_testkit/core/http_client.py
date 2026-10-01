"""Shared HTTP transport for API clients."""

from types import TracebackType
from typing import Any, Self

import requests


class HttpClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    def request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self._timeout)
        # A redirect is part of the observed API contract, not a successful response.
        kwargs.setdefault("allow_redirects", False)
        return self._session.request(
            method=method,
            url=f"{self._base_url}/{path.lstrip('/')}",
            **kwargs,
        )

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
