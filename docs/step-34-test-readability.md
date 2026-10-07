# Step 34: fixture boundaries and readable test scenarios

Test functions should make the action and expected outcome easy to find.
Framework setup, extensive input tables, fake services and embedded child-test
source code previously obscured those outcomes. This refactoring separates those
responsibilities and removes redundant unit combinations.

## Where code belongs

```text
automation/
  src/
    llm_testkit/                # Transport, clients, pages, acceptance and evidence logic
    test_support/
      fixtures/                # Pytest dependencies, patching, setup and cleanup
      builders/                # Object construction, test doubles and scenario preparation
      data/                    # Named Python parameter cases and offline inputs
        scripts/               # Synthetic child-pytest modules and configuration templates
  test_data/                   # Source-bound documents, reviewed datasets and acceptance profiles
  tests/
    conftest.py                # Plugin registration and assertion rewriting
    unit/                      # Framework behavior, without application requests
    browser/                   # Page Object regressions against local HTML
    ui/                        # Application browser scenarios
    conversation/              # Chat-mode acceptance scenarios
```

`test_support` is an installed package within the automation project, so imports
work the same way in the terminal, IDE and CI. Builders and Python parameter data
are test scaffolding. `test_data` holds reviewed application expectations and
documents used across API, UI and evaluation scenarios. Keeping those roles
distinct avoids accidentally treating a mocked response as application evidence.

Fixture implementations are always in `fixtures/`. Shared fixtures are registered
as pytest plugins. Narrowly scoped overrides are explicitly re-exported by a
suite's `conftest.py` or its test module. For example, the adversarial policy-file
override applies only to adversarial tests; registering it globally would alter
ordinary golden cases. Re-exporting an existing fixture does not create another
fixture implementation.

## Reading a scenario

The health performance test now consists of two operations:

```python
report = run_batch(health_workload, **performance_budget)
record_batch(report, performance_report_directory, **health_performance_context)
```

The workload fixture owns each request's transport and checks its response.
Budget and reporting-context fixtures supply explicit configuration. The existing
batch recorder preserves attempts and enforces latency/error thresholds. RAG
workloads also retain model, policy and golden-dataset identity. Setup and indexing
remain outside the timed request, and concurrent requests own separate sessions.

An offline browser scenario reads as preparation followed by a check:

```python
chat_simulation.show_answer_with_delayed_content()
assertions.assert_ui_completed_answer(workspace, expected_text="Ready")
```

JavaScript and local DOM construction live in test support. This scenario verifies
that the Page Object waits for nonempty content; it does not evaluate a model.
Application UI tests still perform real Page Object actions against the application.

Parameter data is imported by name:

```python
@pytest.mark.parametrize("variable,value", INVALID_URL_CASES)
def test_invalid_urls_fail_with_setting_name(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match=variable):
        Settings.from_env()
```

Parameter tables can carry readable `pytest.param(..., id=...)` labels and expected
values. A valid-input check and an invalid-input check use separate test bodies
when their expectations differ. There is no hidden success branch inside a
negative test.

## Choosing useful unit cases

The offline unit suite changed from 562 to 527 collected checks:

| Area | Before | After | Selection rationale |
| --- | ---: | ---: | --- |
| Invalid URLs | 20 | 10 | Both URLs use one validator; each invalid URL shape is checked once, with both setting names represented. |
| Invalid timeouts | 12 | 8 | Every independently guarded field retains zero and NaN cases; negative and infinite values are representative additional inputs. |
| Failed quality gates | 16 | 7 | Every metric retains a below-threshold check; shared missing/invalid/unmeasured paths use one representative metric. All four threshold-boundary checks also remain. |
| Golden-reference assertion | 16 | 4 | The dataset loader validates all 16 references; the shared answer assertion is exercised once per acceptance category. |

The application golden suite still collects all 16 cases against both default
models: 32 checks. Provenance, malformed evidence, failure propagation, resource
cleanup and qualification checks remain. There is no target test count: keep a
case when it exercises a distinct behavior or acceptance rule.

At this stage, unit bodies still used native assertions and `pytest.raises`.
[Step 35](step-35-unit-scenario-layers.md) subsequently moves those checks into
independent `test_support.assertions` modules. Application API/UI checks use
`llm_testkit.assertions`. Conditionals remain appropriate inside validators,
builders, fixtures and reusable checks.

Optional judge/browser imports are lazy in supporting code, so a base installation
can collect tests without installing every extra. Tests contain no local imports,
fixture definitions, nested helper definitions, inline parameter lists or control
flow statements. This was checked by inspecting the test-module syntax trees.

## Verification

Run from `automation` with the appropriate dependencies installed:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest tests/unit -q
python -m pytest tests/browser --run-ui -q
python -m pytest tests/test_golden_rag.py --collect-only -q
```

Local verification passed 527 unit checks, four offline browser checks, six
health/authentication/installation checks and one two-request health performance
batch. The two saved-report tests correctly skip without explicit report inputs.
Statement and branch coverage across `llm_testkit` and fixture implementations is
79.75%, above the unchanged 75% CI floor. Configuration coverage remains 100%;
quality-gate coverage remains 97.33%. Builders and parameter tables are excluded
from the measurement because they are test scaffolding, while resource-owning
fixtures remain included.

Live LLM generation was not rerun for this structural change. Collection confirms
the golden model matrix is unchanged; it does not prove that today's generated
answers meet every acceptance rule.

## Adding another test

1. Put reusable acceptance inputs in `test_data` and unit parameter rows in the
   relevant `test_support/data` module.
2. Put mutable setup and resource ownership in a scoped fixture. Register cleanup
   as soon as creation supplies a usable identifier, before later validation.
3. Give reusable preparation or simulation a name describing its action in
   `builders/`. Avoid helpers named after a test's line number.
4. Write a linear test body with explicit actions and checks. Keep loops,
   branching and optional dependency imports in supporting code.
5. Add a readable reporting title. Run the affected suite and the required static
   checks; use live model calls when the changed behavior actually needs them.
