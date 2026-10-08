# Step 18: Fast framework checks in GitHub Actions

The `Framework checks` workflow runs on every push, pull request and manual
execution. Ubuntu 24.04 jobs use Python 3.12.3 for Ruff, the offline unit suite
and a separate offline Chromium Page Object suite. They do not start AnythingLLM
or Ollama, and require no API keys or repository secrets.

## What is checked

Ruff uses the project's pinned version and an explicit configuration. Its initial
rule set checks common Python errors, undefined names, unused imports, Bugbear
rules and unnecessary trailing commas (`E4`, `E7`, `E9`, `F`, `B`, `COM819`).
The formatting step checks import order with isort and compact layout with YAPF,
using the shared 120-character configuration. Assertion registration precedes
loading the fixture plugins; no import-order exception is needed.

The workflow selects `tests/unit` directly. Evaluation dependencies are installed
so that the RAGAS adapter tests run instead of being skipped; those tests use
mocked judge responses. Node.js on the runner also executes the preload-hook unit
check against a fake SDK. There are no model downloads or answer generations.
Automatic pytest plugin loading is disabled to match the intended offline scope.

The browser job installs the UI dependency profile and Chromium, then selects
`tests/browser --run-ui`. These tests use deterministic local HTML to exercise
Page Objects and answer waits. Allure and evaluation dependencies are unnecessary.

Dependencies and third-party Action revisions are pinned. Pip caches installed
package downloads between runs. Only repository read permission is granted, and
checkout does not retain Git credentials. A newer push on the same branch cancels
an older unfinished run.

## Reports and failures

Open the repository's **Actions** tab, select **Framework checks**, then open a run.
The job shows separate Ruff and unit-test steps. Unit tests still run if Ruff fails,
provided dependency installation succeeded; either failed check fails the job.

The `framework-check-reports` artifact is retained for 14 days and contains:

- `ruff.txt`: static-check output.
- `format.txt`: formatting-check output.
- `pytest.txt`: unit-test output.
- `unit-tests.xml`: JUnit results for tooling and detailed investigation.

The separate `browser-check-reports` artifact contains `browser.xml` and browser
traces/screenshots retained on failure, also for 14 days.

Artifact upload is attempted even when checks fail. Setup failures may leave no
report files; their cause is shown in the installation or setup step. Reports and
runtime evidence remain ignored in the working tree.

## Run the same checks locally

From `automation`, using the existing Python 3.12 environment:

```bash
python -m pip install -r requirements.lock -r requirements-evaluation.lock -r requirements-dev.lock
python -m pip install --no-deps -e .
python -m pip check
python -m ruff check src tests --config pyproject.toml
python -m isort --check-only --diff src tests
python -m yapf --diff --recursive src tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 RAGAS_DO_NOT_TRACK=true LANGCHAIN_TRACING_V2=false \
  python -m pytest tests/unit --junitxml=reports/ci/unit-tests.xml
```

API, UI and live RAG checks remain explicit local runs. A future integration job
can prepare the application separately without expanding this fast framework job.

## Official references

- [GitHub: building and testing Python](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
- [Ruff configuration](https://docs.astral.sh/ruff/configuration/)
