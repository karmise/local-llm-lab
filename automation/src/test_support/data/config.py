"""Named parameter cases for config scenarios."""

INVALID_URLS_VARIABLE_CASES = ["ANYTHINGLLM_BASE_URL", "OLLAMA_BASE_URL"]

INVALID_URLS_VALUE_CASES = [
    "file:///tmp",
    "http://",
    "http://user:secret@localhost",
    "http://localhost?x=1",
    "http://localhost#fragment",
    "http://localhost:bad",
    "http://localhost:65536",
    "http://localhost:0",
    "http://local host",
    "http://[broken",
]

INVALID_TIMEOUTS_VARIABLE_CASES = [
    "ANYTHINGLLM_HTTP_TIMEOUT",
    "ANYTHINGLLM_DOCUMENT_TIMEOUT",
    "ANYTHINGLLM_LLM_TIMEOUT",
]

INVALID_TIMEOUTS_VALUE_CASES = ["0", "-1", "nan", "inf"]
