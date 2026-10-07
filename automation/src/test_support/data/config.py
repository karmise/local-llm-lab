"""Representative setting errors: every URL shape and every timeout field."""

import pytest

INVALID_URL_CASES = [
    pytest.param("ANYTHINGLLM_BASE_URL", "file:///tmp", id="unsupported-scheme"),
    pytest.param("OLLAMA_BASE_URL", "http://", id="missing-host"),
    pytest.param("ANYTHINGLLM_BASE_URL", "http://user:secret@localhost", id="credentials"),
    pytest.param("OLLAMA_BASE_URL", "http://localhost?x=1", id="query"),
    pytest.param("ANYTHINGLLM_BASE_URL", "http://localhost#fragment", id="fragment"),
    pytest.param("OLLAMA_BASE_URL", "http://localhost:bad", id="nonnumeric-port"),
    pytest.param("ANYTHINGLLM_BASE_URL", "http://localhost:65536", id="port-overflow"),
    pytest.param("OLLAMA_BASE_URL", "http://localhost:0", id="zero-port"),
    pytest.param("ANYTHINGLLM_BASE_URL", "http://local host", id="whitespace"),
    pytest.param("OLLAMA_BASE_URL", "http://[broken", id="malformed-ipv6"),
]

# Timeouts have separate field guards, so zero and NaN are checked for every field.
INVALID_TIMEOUT_CASES = [
    pytest.param("ANYTHINGLLM_HTTP_TIMEOUT", "0", id="http-zero"),
    pytest.param("ANYTHINGLLM_DOCUMENT_TIMEOUT", "0", id="document-zero"),
    pytest.param("ANYTHINGLLM_LLM_TIMEOUT", "0", id="llm-zero"),
    pytest.param("ANYTHINGLLM_HTTP_TIMEOUT", "nan", id="http-nan"),
    pytest.param("ANYTHINGLLM_DOCUMENT_TIMEOUT", "nan", id="document-nan"),
    pytest.param("ANYTHINGLLM_LLM_TIMEOUT", "nan", id="llm-nan"),
    pytest.param("ANYTHINGLLM_HTTP_TIMEOUT", "-1", id="negative"),
    pytest.param("ANYTHINGLLM_LLM_TIMEOUT", "inf", id="infinite"),
]
