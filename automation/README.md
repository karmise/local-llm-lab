# Automation framework

New to the project? Start with the [one-hour onboarding guide](../docs/framework-onboarding.md)
for architecture, test strategy, design rationale and a complete scenario walkthrough.

A [Russian translation](../docs/framework-onboarding.ru.md) is also available.

Use Python 3.12 from this directory. The project separates HTTP transport, API
operations, browser actions, assertions, pytest fixtures and reporting. Test
functions describe a scenario; setup and cleanup belong to fixtures.

## Install

```bash
python -m pip install -r requirements.lock -r requirements-dev.lock
python -m pip install --no-deps -e .
```

Install `requirements-evaluation.lock` for the RAGAS adapter unit tests,
`requirements-ui.lock` plus `python -m playwright install chromium` for browser
tests, and `requirements-reporting.lock` for Allure. UI checks can run without
Allure; live/offline quality-report scenarios require it.

## Select a test layer

Run commands from `automation` with its virtual environment activated.

| Purpose | Command | External requirements |
| --- | --- | --- |
| Framework unit tests | `python -m pytest tests/unit -q` | No running services; Node for the capture-hook test |
| Static checks | `python -m ruff check src tests` | Development dependencies |
| Formatting check | `python -m ruff format --check src tests` | Development dependencies |
| Offline Page Object regressions | `python -m pytest tests/browser --run-ui` | Chromium; no application/models |
| Application health and API | `python -m pytest -m 'smoke or api'` | AnythingLLM; API key; embedding model for indexing |
| RAG answers | `python -m pytest tests/test_rag.py --rag-model qwen3.5:4b` | AnythingLLM, API key, generation and embedding models |
| Golden policy dataset (one case) | `python -m pytest tests/test_golden_rag.py --run-golden --rag-model qwen3.5:4b -k carryover_limit` | Same as RAG; broader catalog is opt-in |
| Application UI | `python -m pytest tests/ui --run-ui` | Same as RAG, plus Chromium |

`python -m pytest` retains the existing behavior: it includes live API and RAG
tests. Use `tests/unit` or `-m unit` for an offline run. Browser, golden and live-quality
tests skip until explicitly enabled. Ordinary RAG tests use both configured
default models when `--rag-model` is omitted; repeat the flag to select several.
`--rag-repeat N` creates independent workspaces for each repetition. UI tests use
the workspace-template model and are not expanded by `--rag-model`.

For browser diagnostics, add:

```bash
--tracing retain-on-failure --screenshot only-on-failure \
  --output reports/my-run/browser --alluredir reports/my-run/allure
```

Use a fresh output directory for each run. Omit `--alluredir` without reporting
dependencies. The application UI fixture attaches retained browser artifacts to
Allure when its reporter is enabled.

See the [live quality guide](../docs/step-15-live-quality.md) for the explicit
one-answer/two-judge-call scenario, and the
[offline quality guide](../docs/step-14-allure-quality-report.md) for saved evidence.

## Golden acceptance dataset

The [golden dataset guide](../docs/step-22-golden-dataset.md) describes 16 source-bound
cases covering policy facts, multi-part questions, boundaries and missing information.
Cases have typed, validated references, acceptance rules and source anchors. Their
version and checksum are recorded with model/configuration metadata.

Golden scenarios skip unless `--run-golden` is supplied. A full run uses 16
generations on one explicitly selected model, or 32 with the default two models,
before repetitions. Start with one case. These are curated policy acceptance
checks, not a statistical model-accuracy benchmark or semantic correctness metric.

## Structure and extension rules

| Location | Responsibility |
| --- | --- |
| `src/llm_testkit/core/` | HTTP session lifetime, timeouts, no implicit redirects or retries |
| `src/llm_testkit/clients/` | API routes, request payloads and file uploads; return raw responses |
| `src/llm_testkit/pages/` | Browser locators and actions |
| `src/llm_testkit/assertions.py` | API/answer/source/quality/UI acceptance criteria |
| `src/llm_testkit/pytest_support/options.py` | CLI selection, model matrix, opt-in validation after `-k`/`-m` |
| `src/llm_testkit/pytest_support/environment.py` | Paths, configuration, clients and scenario data |
| `src/llm_testkit/pytest_support/resources.py` | Temporary workspace, document folder, upload and indexing |
| `src/llm_testkit/pytest_support/rag.py` | Model metadata, generation and optional context capture |
| `src/llm_testkit/datasets/` | Typed, source-bound golden dataset loading and validation |
| `test_data/` | Fictional documents and shared question/reference/fact profiles |
| `tests/unit/` | Framework behavior with HTTP calls mocked |
| `tests/browser/` | Real browser with deterministic HTML; no model generation |
| `tests/ui/` | End-to-end application browser scenarios |

Keep fixtures function-scoped for mutable resources. Register cleanup as soon as
a successful create response provides a usable resource identifier, before
validating other response fields. Each cleanup also verifies absence. Pytest
reports test and teardown failures separately and still runs the remaining
finalizers. A failed upload/indexing/check must not leave a workspace behind.

Use plain pytest `assert` in unit tests to verify the framework independently of
its own assertion helpers. Unit tests block Requests calls unless explicitly
mocked; telemetry settings are scoped with `monkeypatch`. Lifecycle and CLI tests
exercise real child pytest runs, not private fixture wrappers or `sys.modules`.

Put shared questions, references and acceptance patterns in `test_data`. API, UI
and quality scenarios should consume the same profiles. `rag_chat` and
`workspace_page` declare their metadata/setup dependencies directly.

Use Playwright's retrying expectations instead of fixed sleeps. After sending a
question, the Page Object targets the next persisted assistant reply; completion
does not require a Sources button. Check source presence separately when the
scenario requires a citation. Keep model retries disabled so failures remain
visible and generation counts stay predictable.

## Known boundaries

- Message/source DOM selectors and capture parsing target AnythingLLM 1.16.2.
  Revalidate them against the real application when upgrading it.
- Required-fact patterns and source fragments are smoke checks, not a semantic
  proof: negations or contradictory statements can still contain matching facts.
- Faithfulness remains an exploratory measurement with no calibrated pass/fail
  threshold. Passing these tests is not a statistical estimate of model accuracy.
- Resource creation with a network timeout or an unusable/missing returned slug
  cannot guarantee cleanup: ownership cannot be established from that response.
- Capture supports one non-streaming answer per temporary workspace; the UI uses
  its own streaming flow. Parallel live model/judge runs have not been validated.

The current changes and validation are recorded in the
[refactoring notes](../docs/step-21-framework-refactoring.md).
