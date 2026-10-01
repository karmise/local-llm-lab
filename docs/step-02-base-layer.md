# Step 2: Layered API Framework

The original availability test now uses separate framework layers. Its behavior
is unchanged: GET `/api/ping`, HTTP 200 and `online=true`. Authentication and
operations under developer API `/api/v1` will follow.

## Structure

Paths are relative to `automation`:

```text
src/llm_testkit/
├── config.py                         # environment settings
├── core/
│   └── http_client.py                # shared HTTP transport
└── clients/
    └── anythingllm_client.py         # AnythingLLM operations
tests/
├── conftest.py                      # client lifecycle and dependencies
└── test_health.py                   # scenario and expectations
```

`__init__.py` files identify Python packages. The `src` layout separates the
framework library from tests and data. `pyproject.toml` configures package
discovery in `src` and pytest's `importlib` mode. Imports use the installed
`llm_testkit` package without modifying `sys.path`.

## Responsibilities

| Layer | Responsibility |
| --- | --- |
| `Settings` | Read and validate the application URL and request timeout |
| `HttpClient` | Build URLs, send requests through `requests.Session`, apply timeouts |
| `AnythingLLMClient` | Expose named application operations, currently `health()` |
| Fixtures | Construct dependencies and release HTTP sessions |
| Tests | Call operations and assert response status and content |

The application client receives its transport through constructor injection.
This uses composition without requiring a class hierarchy or `BaseTest`.
Workspace and document operations can later be split into resource clients if
the application client grows.

## Request lifecycle

1. pytest resolves the test's `anythingllm_api` fixture.
2. That fixture depends on `http_client`, which depends on `settings`.
3. `Settings.from_env()` loads environment variables.
4. `AnythingLLMClient.health()` supplies GET and `/api/ping` to the transport.
5. `HttpClient` returns a `requests.Response`.
6. The test validates the response. Fixture teardown closes the session even
   when the test fails.

Immutable settings are read once per pytest session. Clients are created per
test so cookies and session state do not leak between scenarios. Requests within
one test share a session.

The transport does not call `raise_for_status()`: negative tests need to inspect
responses such as an expected HTTP 401. Network and JSON parsing errors remain
visible. Automatic retries are not enabled; retrying writes can duplicate data
and retrying assertions can hide instability.

## Installation and execution

From `local-llm-lab/automation`, with AnythingLLM running:

```bash
source .venv/bin/activate
python -m pip install --no-deps -e .
python -m pytest -m smoke -v
```

Editable installation links the environment to `src`; changes to Python source
files do not require package reinstallation. When creating a fresh environment,
install `requirements.lock` first, as described in step 1.

```bash
ANYTHINGLLM_BASE_URL=http://127.0.0.1:3001 ANYTHINGLLM_HTTP_TIMEOUT=10 python -m pytest -v
```

Defaults: URL `http://127.0.0.1:3001`, timeout five seconds. A Requests timeout
limits connection and read waits, not the total duration of a scenario. Model
generation will need a separate operation timeout. `/api/ping` does not require
a developer API key.

In PyCharm, use `automation/.venv/bin/python` and the `automation` working
directory. If package imports fail, check the interpreter and editable install.

Verified on 2026-10-01 against local AnythingLLM: `1 passed in 0.02s`.
`pip check` found no dependency conflicts. The local report is
`automation/reports/step-02.xml`, relative to the environment root. This verifies
availability through the new layers, not generation or RAG behavior.

## Reading order and next step

Read `tests/test_health.py`, `tests/conftest.py`,
`clients/anythingllm_client.py`, `core/http_client.py`, then `config.py`.
Follow how pytest supplies the client and who closes its session.

Next: developer API authentication and the first workspace operation. UI Page
Objects, RAG evaluations, reporting and CI will follow their respective scenarios.

## Official references

- [pytest project layout](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [Requests sessions](https://requests.readthedocs.io/en/latest/user/advanced/#session-objects)
- [Requests timeouts](https://requests.readthedocs.io/en/latest/user/advanced/#timeouts)
- [AnythingLLM developer API](https://docs.anythingllm.com/features/api)
