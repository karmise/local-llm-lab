# Automation framework

New to the project? Start with the [one-hour onboarding guide](../docs/framework-onboarding.md)
for architecture, test strategy, design rationale and a complete scenario walkthrough.

A [Russian translation](../docs/framework-onboarding.ru.md) is also available.

Use Python 3.12 from this directory. The project separates HTTP transport, API
operations, browser actions, assertions, pytest fixtures and reporting. Test
functions describe a scenario; setup and cleanup belong to fixtures.

Python code uses a compact style with a 120-character line limit. Imports, decorators,
signatures and calls stay on one line when they fit; longer argument lists wrap
without placing each argument or closing parenthesis on its own line. Optional
trailing commas are omitted; commas required for singleton tuples are preserved.
YAPF formats code, isort orders imports, and Ruff checks Python errors and style.
The shared configuration is in `pyproject.toml`; local formatter exceptions are
not needed.

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
| Import order check | `python -m isort --check-only --diff src tests` | Development dependencies |
| Formatting check | `python -m yapf --diff --recursive src tests` | Development dependencies |
| Offline Page Object regressions | `python -m pytest tests/browser --run-ui` | Chromium; no application/models |
| Application health and API | `python -m pytest -m 'smoke or api'` | AnythingLLM; API key; embedding model for indexing |
| API contract and negative cases | `python -m pytest tests/api/test_api_contract.py` | AnythingLLM; API key; no model calls |
| RAG answers | `python -m pytest tests/rag/test_rag.py --rag-model qwen3.5:4b` | AnythingLLM, API key, generation and embedding models |
| Golden policy dataset (one case) | `python -m pytest tests/rag/test_golden_rag.py --run-golden --rag-model qwen3.5:4b -k carryover_limit` | Same as RAG; broader catalog is opt-in |
| Application UI | `python -m pytest tests/ui --run-ui` | Same as RAG, plus Chromium |
| Conversational assistant | `python -m pytest tests/rag/conversation --run-conversation --rag-model qwen3.5:4b` | AnythingLLM, API key, generation and embedding models; five sequential generations |

To apply the shared formatting rules locally:

```bash
python -m isort src tests
python -m yapf --in-place --recursive src tests
```

Use YAPF for automatic formatting in the editor as well; running `ruff format`
would apply a different layout. Ruff remains the linter.

`python -m pytest` retains the existing behavior: it includes live API and RAG
tests. Use `tests/unit` or `-m unit` for an offline run. Browser, golden and live-quality
tests skip until explicitly enabled. Ordinary RAG tests use both configured
default models when `--rag-model` is omitted; repeat the flag to select several.
`--rag-repeat N` creates independent workspaces for each repetition. UI tests use
the workspace-template model and are not expanded by `--rag-model`.

Conversation checks use the separate Chat profile and skip without
`--run-conversation`. Their default is one model (`qwen3.5:4b`); an explicit model
matrix and independent repetitions are supported. See the
[conversation guide](../docs/evaluation-guide.md#conversation-profile) for the contract and scope.

For browser diagnostics, add:

```bash
--tracing retain-on-failure --screenshot only-on-failure \
  --output reports/my-run/browser --alluredir reports/my-run/allure
```

Use a fresh output directory for each run. Omit `--alluredir` without reporting
dependencies. The application UI fixture attaches retained browser artifacts to
Allure when its reporter is enabled.

See the [live quality guide](../docs/evaluation-guide.md#combined-quality-report-and-gates) for the explicit
one-answer/two-judge-call scenario, and the
[offline quality guide](../docs/evaluation-guide.md#combined-quality-report-and-gates) for saved evidence.

## Golden acceptance dataset

The [golden dataset guide](../docs/evaluation-guide.md#golden-dataset) describes 16 source-bound
cases covering policy facts, multi-part questions, boundaries and missing information.
Cases have typed, validated references, acceptance rules and source anchors. Their
version and checksum are recorded with model/configuration metadata.

Golden scenarios skip unless `--run-golden` is supplied. A full run uses 16
generations on one explicitly selected model, or 32 with the default two models,
before repetitions. Start with one case. These are curated policy acceptance
checks, not a statistical model-accuracy benchmark or semantic correctness metric.

## Factual correctness evaluation

The [correctness guide](../docs/evaluation-guide.md#semantic-metrics-on-a-captured-sample) describes reference-based
RAGAS factual F1 with four bounded local judge calls and labelled controls. It is
explicitly invoked for a captured golden case and can be added as optional saved
evidence with `--correctness-report`. Existing live-quality budgets are unchanged.
Correctness remains a measurement until explicit gates are supplied. Context precision/recall can be evaluated separately as described below.

## Structure and extension rules

| Location | Responsibility |
| --- | --- |
| `src/llm_testkit/core/` | HTTP session lifetime, timeouts, no implicit redirects or retries |
| `src/llm_testkit/clients/` | `BaseClient` (base path, API key, path encoding) and the API clients built on it: `AnythingLLMClient` with `.workspaces` and `.documents`, and `OllamaClient`; they return raw responses |
| `src/llm_testkit/pages/` | `BasePage` (URL, navigation, timeouts) and `WorkspacePage` (locators and actions) |
| `src/llm_testkit/assertions/` | Acceptance criteria by area: `fields`, `api`, `answers`, `quality`, `ui` |
| `src/llm_testkit/pytest_support/options.py` | CLI selection, model matrix, opt-in validation after `-k`/`-m` |
| `src/test_support/fixtures/` | Scoped setup, dependencies, resource ownership, evidence and cleanup |
| `src/test_support/builders/` | Public, per-domain test data builders and deterministic doubles shared by test modules |
| `src/llm_testkit/datasets/` | Typed, source-bound golden dataset loading and validation |
| `test_data/` | Fictional documents and shared question/reference/fact profiles |
| `tests/` | Test suites grouped by level; see the layout below |

### Test layout

Suites are grouped by what they need to run, and then by feature:

| Directory | Level | Needs | Contents |
| --- | --- | --- | --- |
| `tests/unit/` | Framework unit | Nothing; HTTP is blocked | Mirrors `src/llm_testkit`: `core`, `assertions`, `datasets`, `evaluation`, `reporting`, `ci`, `qualification`, `performance`, `pytest_support` |
| `tests/browser/` | Page Object | Chromium and local HTML | Waiting and selection rules of the Page Object |
| `tests/environment/` | Installation (IQ) | AnythingLLM, Ollama | Runtime versions, declared models, health |
| `tests/api/` | API, no generation | AnythingLLM, API key | Authentication, API contract and negative cases, workspace and document lifecycle |
| `tests/rag/` | Generated answers | Plus generation and embedding models | Answers and sources, golden dataset, `conversation/` (Chat mode) |
| `tests/regression/` | Changed inputs, same criteria | As RAG, opt-in | Prompt regression, adversarial inputs, counterfactual bias |
| `tests/ui/` | End-to-end UI | As RAG, plus Chromium | Chat, sources and history in the real application |
| `tests/quality/` | Evidence and gates | Saved evidence; live quality also needs the capture overlay | Quality report, live quality, benchmark and CI benchmark reports |
| `tests/performance/` | Load | As API or RAG, opt-in | Bounded health and RAG batches |

Positive and negative cases of a feature stay in the same module, so the module
reads as that feature's specification. Directory-scoped `conftest.py` files keep
fixtures local: the Chat workspace profile applies only under `rag/conversation/`,
and the UI teardown hook only under `ui/`.

Keep fixtures function-scoped for mutable resources. Register cleanup as soon as
a successful create response provides a usable resource identifier, before
validating other response fields. Each cleanup also verifies absence. Pytest
reports test and teardown failures separately and still runs the remaining
finalizers. A failed upload/indexing/check must not leave a workspace behind.

Unit tests use plain pytest ([why](../docs/decisions.md#unit-tests-are-plain-pytest-verified-by-mutation-testing)).
Each test shows its inputs, action and expected values; a rejection names the rule
it enforces with `pytest.raises(..., match=...)` and has a paired acceptance test.
Data shared by several test modules comes from public builders in
`test_support/builders/<domain>.py`; data, fixtures and child-test sources used by
one module live in that module. Plain `assert` keeps unit checks independent of
application Assertions. Mutation testing (mutmut) measured each domain's tests.
Unit tests block Requests calls unless explicitly mocked; telemetry settings are
scoped with `monkeypatch`. Lifecycle and CLI tests exercise real child pytest runs.

Put shared questions, references and acceptance patterns in `test_data`. API, UI
and quality scenarios should consume the same profiles. `rag_chat` and
`workspace_page` declare their metadata/setup dependencies directly.

Register shared fixture plugins for live and browser suites in `tests/conftest.py`;
re-export narrowly scoped fixtures from the relevant suite so an adversarial document
override cannot affect ordinary RAG tests. Unit tests use inline `pytest.param(...,
id=...)` tables. Keep scenario bodies linear: arrange, act, assert. Conditional setup, test doubles and browser simulation
belong to test support; acceptance rules belong to `Assertions`. Optional browser
and judge imports remain lazy in supporting code.

Builders return a fresh copy on every call, so a test may edit what it receives.
Protocol field names and meaningful expected values stay explicit in checks.

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
- Semantic thresholds are experimental, not calibrated; factual correctness has
  none. Passing these tests is not a statistical estimate of model accuracy.
- Resource creation with a network timeout or an unusable/missing returned slug
  cannot guarantee cleanup: ownership cannot be established from that response.
- Capture supports one non-streaming answer per temporary workspace; the UI uses
  its own streaming flow. Parallel live model/judge runs have not been validated.

## Opt-in evaluation suites

Capture, semantic metrics and judge controls, quality reports and gates, golden
and regression suites, the dataset benchmark, fresh AI CI, metric history,
performance and qualification evidence are described, with commands, in the
[evaluation guide](../docs/evaluation-guide.md). The reasons behind the framework's
rules are in [decisions](../docs/decisions.md).
