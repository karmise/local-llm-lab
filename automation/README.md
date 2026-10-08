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
| RAG answers | `python -m pytest tests/test_rag.py --rag-model qwen3.5:4b` | AnythingLLM, API key, generation and embedding models |
| Golden policy dataset (one case) | `python -m pytest tests/test_golden_rag.py --run-golden --rag-model qwen3.5:4b -k carryover_limit` | Same as RAG; broader catalog is opt-in |
| Application UI | `python -m pytest tests/ui --run-ui` | Same as RAG, plus Chromium |
| Conversational assistant | `python -m pytest tests/conversation --run-conversation --rag-model qwen3.5:4b` | AnythingLLM, API key, generation and embedding models; five sequential generations |

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
[conversation guide](../docs/step-32-conversation.md) for the contract and scope.

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

## Factual correctness evaluation

The [correctness guide](../docs/step-23-factual-correctness.md) describes reference-based
RAGAS factual F1 with four bounded local judge calls and labelled controls. It is
explicitly invoked for a captured golden case and can be added as optional saved
evidence with `--correctness-report`. Existing live-quality budgets are unchanged.
Correctness remains a measurement until explicit gates are supplied. Context precision/recall can be evaluated separately as described below.

## Structure and extension rules

| Location | Responsibility |
| --- | --- |
| `src/llm_testkit/core/` | HTTP session lifetime, timeouts, no implicit redirects or retries |
| `src/llm_testkit/clients/` | API routes, request payloads and file uploads; return raw responses |
| `src/llm_testkit/pages/` | Browser locators and actions |
| `src/llm_testkit/assertions.py` | API/answer/source/quality/UI acceptance criteria |
| `src/test_support/assertions/` | Independent unit expectations: values, exceptions, mock calls and domain evidence |
| `src/llm_testkit/pytest_support/options.py` | CLI selection, model matrix, opt-in validation after `-k`/`-m` |
| `src/test_support/fixtures/` | Scoped setup, dependencies, resource ownership, evidence and cleanup |
| `src/test_support/builders/` | Test objects, deterministic doubles and named scenario preparation |
| `src/test_support/data/` | Named parameter cases and offline test inputs |
| `src/test_support/data/scripts/` | Isolated pytest source templates for collection/lifecycle checks |
| `src/llm_testkit/datasets/` | Typed, source-bound golden dataset loading and validation |
| `test_data/` | Fictional documents and shared question/reference/fact profiles |
| `tests/unit/` | Framework behavior with HTTP calls mocked |
| `tests/browser/` | Real browser with deterministic HTML; no model generation |
| `tests/ui/` | End-to-end application browser scenarios |
| `tests/conversation/` | Opt-in Chat behavior with a scoped conversational workspace profile |

Keep fixtures function-scoped for mutable resources. Register cleanup as soon as
a successful create response provides a usable resource identifier, before
validating other response fields. Each cleanup also verifies absence. Pytest
reports test and teardown failures separately and still runs the remaining
finalizers. A failed upload/indexing/check must not leave a workspace behind.

Use `test_support.assertions` for unit expectations. Its native pytest checks are
independent of application Assertions, so a broken application check cannot
verify its own output. Exception helpers preserve type, regex matching and the
original captured exception. Shared checks are included in the CI coverage gate.
Unit tests block Requests calls unless explicitly mocked; telemetry settings are
scoped with `monkeypatch`. Lifecycle and CLI tests exercise real child pytest runs.

Put shared questions, references and acceptance patterns in `test_data`. API, UI
and quality scenarios should consume the same profiles. `rag_chat` and
`workspace_page` declare their metadata/setup dependencies directly.

Keep fixtures out of test modules. Register shared fixture plugins in
`tests/conftest.py`; re-export narrowly scoped fixtures from the relevant suite
or module so an adversarial document override cannot affect ordinary RAG tests.
Use imported parameter tables rather than inline lists. Keep scenario bodies
linear: prepare, act, check. Conditional setup, test doubles and browser simulation
belong to test support; acceptance rules belong to `Assertions`. Optional browser
and judge imports remain lazy in supporting code.

Prefer prepared fixtures for a stable scenario, such as `correctness_judge` or
`correctness_evidence`. Use a fixture factory when a test needs several clients,
responses or configurations. Mutable input catalogs are copied with
`test_support.data.common.fresh`; never mutate a shared parameter dictionary.
Protocol field names and meaningful expected values can remain explicit in checks.

The [test readability guide](../docs/step-34-test-readability.md) explains these
boundaries and the selection of representative unit cases.
The [shared-check and fixture guide](../docs/step-35-unit-scenario-layers.md)
walks through the subsequent extraction of checks, object construction and payloads.

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
The [second refactoring review (Russian)](../docs/automation-refactor-review.ru.md)
documents report-integrity fixes, shared validation, baseline compatibility and
the latest local verification after the expanded suites were added.

Reference-based retrieval evaluation is described in [Context precision and recall](../docs/step-24-context-relevance.md). It reuses captured samples and adds two independently validated dimensions to the quality report.

[Prompt regression](../docs/step-25-prompt-regression.md) compares versioned system prompts against unchanged golden expectations. The suite is opt-in and the comparison command requires an explicit case/model scope, with missing or uncontrolled runs treated as incomplete.

[Adversarial inputs](../docs/step-26-adversarial-inputs.md) reuse golden policy expectations for six opt-in attacks. Document-injection scenarios require actual captured-context exposure; uploaded citations alone do not count as proof that the model saw an attack.

[Quality gates and coverage](../docs/step-27-quality-gates.md) adds a 75% framework coverage floor in CI and opt-in experimental thresholds for all four saved semantic metrics. A reusable/manual workflow validates a producer's evidence artifact without model calls; local generation is not automatically available on hosted runners.

[Judge suitability and disagreement review](../docs/step-36-judge-validation.md) adds eight source-bound controls for all four metrics, a serial 18-call judge experiment, and offline diagnostics for verified saved benchmarks. The observed reference-attribution mismatch keeps release suitability unestablished; thresholds remain experimental.

[Metric history and baseline monitoring](../docs/step-28-metric-history.md) records immutable, provenance-bound measurements and compares new evidence to an explicit baseline without model calls.

[Performance checks](../docs/step-29-performance.md) provide explicitly selected health or RAG batches, bounded concurrency, individual timing/error evidence and declared latency gates. They do not run model load on ordinary CI pushes.

[Counterfactual bias checks](../docs/step-30-bias.md) compare three employee-descriptor pairs against identical policy acceptance criteria and expose asymmetric, shared-failure or incomplete outcomes.

[Qualification plan and evidence](../docs/step-31-qualification-evidence.md) maps reviewed requirements to test selectors and declared matrices, labels JUnit/Allure results, and builds scoped educational IQ/OQ/PQ packages that retain missing cells and failures.
