# Step 1: Python Project and Application Health Check

## Layout and environment

The environment configuration lives in `local-llm-lab`; the standalone Python
project lives in `local-llm-lab/automation`. Paths below are relative to
`automation`.

- `pyproject.toml`: Python requirements, dependencies and pytest configuration.
- `.venv/`: local virtual environment; excluded from Git.
- `requirements.lock`: pinned runtime dependencies.
- `tests/test_health.py`: application health check.
- `test_data/company-policy.txt`: fictional policy for subsequent RAG checks.
- `reports/`: local test results; excluded from Git.

The existing Python 3.12.3 installation was used. System Python settings were
not changed. Initially the project contained tests only (`packages = []`).
[Step 2](step-02-base-layer.md) adds `src/llm_testkit` and updates package discovery.

## Run locally

From the environment root:

```bash
./lab.sh start
cd automation
source .venv/bin/activate
python -m pytest -v
```

`python -m pytest` uses pytest from the selected interpreter. One test should
pass. `deactivate` exits the virtual environment without stopping AnythingLLM.

Verified on 2026-10-01 with Python 3.12.3, pytest 9.1.1 and requests 2.34.2:
`1 passed in 0.77s`; dependency validation passed. The initial JUnit report is
`automation/reports/step-01.xml`, relative to the environment root. Elapsed pytest
time is not model generation performance.

## Recreate the virtual environment

Install Python 3.12 and prepare the local application environment first. From
`local-llm-lab`:

```bash
cd automation
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pytest -v
```

## Configure PyCharm

Open the environment root or the `automation` directory. Select the existing
`automation/.venv/bin/python` interpreter. The former root `.venv` is no longer
used. Set the test working directory to `automation` and target `tests` or
`tests/test_health.py`.

The environment was recreated after moving the Python project into `automation`;
virtual environment scripts contain absolute interpreter paths. Verification
after relocation: `1 passed in 0.80s`, with `automation/reports/relocation.xml`.

## Health check behavior

1. Use `ANYTHINGLLM_BASE_URL`, defaulting to `http://127.0.0.1:3001`.
2. Send GET `/api/ping` with a five-second Requests timeout.
3. Assert HTTP 200.
4. Assert `online=true` in the JSON response.

```bash
ANYTHINGLLM_BASE_URL=http://127.0.0.1:3001 python -m pytest -m smoke -v
```

This endpoint checks application availability. It does not verify generation,
embeddings, answer quality or a developer API key. An unavailable application
fails the test instead of skipping it.

## Next step

[Build the API framework layers](step-02-base-layer.md): settings, HTTP transport,
application client and fixtures. Developer API authentication follows afterwards.
