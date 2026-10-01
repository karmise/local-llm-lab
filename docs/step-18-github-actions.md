# Step 18: Fast framework checks in GitHub Actions

The `Framework checks` workflow runs on every push, pull request and manual
execution. One Ubuntu 24.04 job uses Python 3.12.3 to run Ruff and the full offline
unit suite. It does not start AnythingLLM, Ollama or a browser, and requires no
API keys or repository secrets.

## What is checked

Ruff uses the project's pinned version and an explicit configuration. Its initial
rule set checks common Python errors, undefined names, unused imports and basic
syntax/style issues (`E4`, `E7`, `E9`, `F`). Formatting and import-order rewrites
are not required at this stage. The fixture module's `E402` exception preserves
pytest assertion registration before importing the shared Assertions module.

The workflow selects `tests/unit` directly. Evaluation dependencies are installed
so that the RAGAS adapter tests run instead of being skipped; those tests use
mocked judge responses. Node.js on the runner also executes the preload-hook unit
check against a fake SDK. There are no model downloads or answer generations.
Automatic pytest plugin loading is disabled to match the intended offline scope.

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
- `pytest.txt`: unit-test output.
- `unit-tests.xml`: JUnit results for tooling and detailed investigation.

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
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 RAGAS_DO_NOT_TRACK=true LANGCHAIN_TRACING_V2=false \
  python -m pytest tests/unit --junitxml=reports/ci/unit-tests.xml
```

API, UI and live RAG checks remain explicit local runs. A future integration job
can prepare the application separately without expanding this fast framework job.

## Official references

- [GitHub: building and testing Python](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
- [Ruff configuration](https://docs.astral.sh/ruff/configuration/)
