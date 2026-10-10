# Step 35: shared expectations and prepared unit scenarios

> Superseded by [step 39](step-39-test-architecture.md): every unit domain now uses
> plain pytest and public builders, and `test_support/assertions` and `test_support/data`
> were removed. This page records the previous structure.

Unit scenarios now separate the operation being tested from object construction,
input payloads and expected-result logic. Test bodies contain readable actions
and calls to reusable checks. There are no direct `assert` statements,
`pytest.raises` blocks, inline dictionaries or direct constructors for the common
clients, judges, mocks and responses in executable test functions.

## The two assertion boundaries

| Location | Purpose |
| --- | --- |
| `llm_testkit/assertions.py` | Application API/UI acceptance, answer facts, citations and evidence gates. |
| `test_support/assertions/values.py` | Equality, identity, membership, size, truth and ordered comparisons. |
| `test_support/assertions/errors.py` | Expected exception type, regex and original exception capture. |
| `test_support/assertions/mocks.py` | Mock call counts and exact arguments. |
| `test_support/assertions/pytest_runs.py` | Outcomes of isolated child pytest runs. |
| `test_support/assertions/judges.py` | Measured scores, exhausted request budgets and retained service failures. |
| Other domain modules in `test_support/assertions/` | Correctness, provenance snapshots, stability, qualification and transport expectations. |

The unit layer uses native pytest assertions internally. It does not delegate value
or exception checks to application Assertions. This matters when testing the
application Assertions themselves: a faulty check must not also be used to confirm
that its own result is correct. Tests may still invoke an application assertion
as the operation under test, then independently verify its result or rejection.

Pytest assertion rewriting is registered for both packages. Moving a check should
retain useful actual/expected diagnostics. Exception helpers return the original
`ExceptionInfo`, including the same exception object and exit code. They execute
the supplied operation inside the exception expectation, preserving lazy argument
evaluation where a callback is used.

A child-pytest contract deliberately supplies a wrong score, wrong exception type
and wrong message, and verifies three failures remain visible. A fourth child case
independently verifies captured exception identity. This guards against a shared
helper silently turning failing tests green.

## A fixture supplies a complete judge scenario

The correctness test reads:

```python
scored = correctness_judge.evaluate()
judge_checks.score_matches(correctness_judge, scored)
judge_checks.budget_is_exhausted(correctness_judge)
```

The parameterized fixture builds a fresh structured-response client and bounded
judge. It supplies the sample, reference verdicts, expected score and request budget.
The scenario evaluates real RAGAS logic against mocked responses. The last check
also confirms that rejecting an extra request does not call the transport again.

Faithfulness and relevance use their own prepared fixtures. Service-failure
fixtures prepare failed judge calls and patched clients; tests verify retained
failure evidence and transport closure. Optional evaluation imports are still
lazy, so collection does not require every optional dependency.

Fixtures in `test_support/fixtures/unit_factories.py` serve tests that need multiple
custom objects: mock/response/client factories, independent settings, failure
objects, XML nodes, an operation lock and a scoped evidence-writer pool. A factory
returns a fresh instance for each call. Client context managers remain explicit
where their enter/exit behavior is the subject of the test. Application invalid-key
authentication also uses a fixture with the existing HTTP transport ownership.

## Evidence mutation is a named action

The scenario titled “Correctness evidence remains bound to application sample and
golden expectations” now has two steps:

```python
correctness_evidence.invalidate(change)
correctness_checks.rejects_unbound_evidence(correctness_evidence)
```

The fixture supplies a fresh sample and evidence pair. The mutation-field table
is in `test_support/data/correctness.py`. The scenario object applies the selected
mutation; the check verifies rejection against the reviewed dataset. No mapping of
field names, synthetic payload construction or exception-matching machinery is
embedded in the test body.

The optional-quality test likewise builds a report, checks its low correctness
measurement, invalidates its sample checksum and verifies that the correctness
dimension becomes an error while faithfulness remains measured. File creation and
checksum preparation belong to its fixture/scenario, and those domain expectations
belong to the correctness check module.

## Data ownership

Static dictionaries live in the corresponding `test_support/data` domain module.
Payloads needing a runtime sample, checksum, temporary path or case are created by
named builders. Shared synthetic model identities and filenames are centralized
in `test_support/data/common.py`. Reviewed application documents and golden
expectations remain in `test_data`.

`common.fresh` makes a deep copy of mutable catalog input before a test changes it.
Fixtures are function-scoped for mutable scenario objects. This prevents a damaged
evidence row in one negative test from changing the next test's starting point.
Names describe the input's purpose, such as `EVIDENCE_MUTATION_FIELDS` or
`DETERMINISTIC_JUDGE_OPTIONS`.

Protocol keys and an expected value central to a test can remain explicit in a
check. Turning every number or field name into a constant would make the expected
behavior harder to see. Shared payloads, repeated identities and lengthy setup
are the parts that need reuse and separate ownership.

## Verification and extension

The refactoring preserves the application golden matrix: all 16 cases still
collect against both default models. The offline unit suite contains 528 checks;
one new check exercises the shared-expectation failure contract. Local verification
also includes four offline browser checks and six health/authentication/installation
checks. Live model generation is not part of this structural verification.

The CI coverage source now includes `test_support.assertions` as well as
`llm_testkit` and fixture implementations. Builders and data catalogs remain test
scaffolding; the 75% coverage floor is unchanged.

When extending a scenario:

1. Put shared inputs in the relevant data module and dynamic construction in a builder.
2. Prefer a prepared fixture for stable setup. Use a fixture factory when comparing
   several fresh instances or configurations is part of the scenario.
3. Add a named domain check for repeated expectations involving several fields.
   Use a typed value or error primitive for a simple expectation.
4. Keep the test body linear and preserve its explicit action and expected outcome.
5. Run the affected suite and required checks; select live generation when behavior
   changes require it.
