"""Test environment settings, independent of pytest and API clients."""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://127.0.0.1:3001"
    ollama_base_url: str = "http://127.0.0.1:11434"
    http_timeout: float = 5.0
    document_timeout: float = 180.0
    llm_timeout: float = 300.0
    api_key: str | None = field(default=None, repr=False)
    workspace_slug: str = "company-policy-lab"

    def __post_init__(self) -> None:
        for address, variable in ((self.base_url, "ANYTHINGLLM_BASE_URL"), (self.ollama_base_url, "OLLAMA_BASE_URL")):
            try:
                url = urlsplit(address)
                port = url.port
            except ValueError:
                raise ValueError(f"{variable} must be an HTTP(S) URL with a valid port") from None
            if url.scheme not in {"http", "https"} or not url.hostname:
                raise ValueError(f"{variable} must be an HTTP(S) URL")
            if any(character.isspace() for character in address) or port == 0:
                raise ValueError(f"{variable} must not contain whitespace or an invalid port")
            if url.query or url.fragment or url.username or url.password:
                raise ValueError(f"{variable} must not contain credentials, query or fragment")
        if not math.isfinite(self.http_timeout) or self.http_timeout <= 0:
            raise ValueError("ANYTHINGLLM_HTTP_TIMEOUT must be a finite positive number")
        if not math.isfinite(self.document_timeout) or self.document_timeout <= 0:
            raise ValueError("ANYTHINGLLM_DOCUMENT_TIMEOUT must be a finite positive number")
        if not math.isfinite(self.llm_timeout) or self.llm_timeout <= 0:
            raise ValueError("ANYTHINGLLM_LLM_TIMEOUT must be a finite positive number")

    @classmethod
    def from_env(cls, default_api_key_file: Path | None = None) -> "Settings":
        api_key = os.getenv("ANYTHINGLLM_API_KEY", "").strip() or None
        key_file = os.getenv("ANYTHINGLLM_API_KEY_FILE")
        if api_key is None:
            path = Path(key_file) if key_file else default_api_key_file
            if path is not None and path.is_file():
                api_key = path.read_text(encoding="utf-8").strip() or None
            elif key_file:
                raise ValueError("ANYTHINGLLM_API_KEY_FILE does not point to an existing file")
        return cls(
                base_url=os.getenv("ANYTHINGLLM_BASE_URL", "http://127.0.0.1:3001").rstrip("/"),
                ollama_base_url=os.getenv("OLLAMA_BASE_URL",
                "http://127.0.0.1:11434").rstrip("/"), http_timeout=float(os.getenv("ANYTHINGLLM_HTTP_TIMEOUT",
                "5")), document_timeout=float(os.getenv("ANYTHINGLLM_DOCUMENT_TIMEOUT",
                "180")), llm_timeout=float(os.getenv("ANYTHINGLLM_LLM_TIMEOUT",
                "300")), api_key=api_key, workspace_slug=os.getenv("ANYTHINGLLM_WORKSPACE_SLUG", "company-policy-lab"))
