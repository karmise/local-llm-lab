# Step 39: readable unit tests verified by mutation testing

This step starts migrating unit tests from the layered scenario style of
[step 35](step-35-unit-scenario-layers.md) to plain pytest tests whose inputs,
action and expectations are visible in one place. The first migrated domain is
factual correctness. Mutation testing measures whether the rewritten tests are
at least as strong as the ones they replace.

## Why the previous structure was changed

Step 35 moved every value, mutation and check out of test bodies into
`test_support/data`, `builders`, `fixtures` and `assertions`. A correctness test read:

```python
@pytest.mark.parametrize(("rv", "gv", "expected"), REAL_RAGAS_CORRECTNESS_WITH_MOCKED_JUDGE_RV_GV_EXPECTED_CASES, ...)
def test_real_ragas_correctness_with_mocked_judge(rv, gv, expected, correctness_judge):
    scored = correctness_judge.evaluate()
    judge_checks.score_matches(correctness_judge, scored)
```

The parameters are unused in the body; the fixture receives them implicitly by name.
Understanding one test required four modules, and string-dispatch builders such as
`prepare_invalid_claim_evidence_is_rejected_case(change, result)` held the actual
test logic. This is the *Obscure Test* smell (Meszaros, *xUnit Test Patterns*).

The indirection also hid weak checks. Mutation testing showed that the old tests
detected only half of the injected faults, and found two tests that passed for the
wrong reason:

- The "raw calls must match summary" test corrupted the raw claims so that they
  failed basic validation. The branch comparing a valid summary with valid raw
  calls was never reached.
- The "incorrect control labels" test produced a score outside the control's F1
  range. The range check rejected it before any label was compared.

Neither checked the error message, so any `ValueError` was accepted.

## Rules for migrated unit tests

1. A test shows arrange, act and assert. Expected values are literals or named
   constants defined in the same module.
2. Parameter tables use `pytest.param(..., id=...)` next to the test. Each negative
   case carries the error it must produce, checked with `pytest.raises(..., match=...)`.
3. Use plain `assert` and `pytest.raises`. Pytest assertion rewriting already gives
   detailed diffs, and plain assertions stay independent of application `Assertions`.
4. Fixtures used by one module live in that module. Fixtures shared across modules
   stay in `test_support/fixtures` and are registered once in `conftest.py`.
5. `test_support/builders/<domain>.py` holds reusable test data builders with
   public names (`make_result`, `make_application_sample`). No test-specific
   `prepare_<test>_case` functions and no imports of private helpers.
6. Every rejection test has a matching acceptance test, so a validator that always
   rejects cannot pass.

Shared doubles that more than one domain needs belong in a common builder; the
Ollama structured-chat reply is now `test_support/builders/ollama.py` instead of
two copies.

## Measured result for correctness

Mutation testing used [mutmut](https://github.com/boxed/mutmut) 3.8 against
`llm_testkit/evaluation/correctness.py`, running only `tests/unit/test_correctness.py`.

| Measurement | Before | After |
| --- | ---: | ---: |
| Tests | 25 | 80 |
| Helper modules for the domain | 4 | 1 builder |
| Mutants detected | 420 / 845 (49.7%) | 777 / 845 (92.0%) |
| Line and branch coverage of the module | 72.6% | 98.5% |

The 68 remaining mutants change error-message capitalisation, CLI help text or a
redundant check (see below). `match=` deliberately matches the meaningful part
of a message, not its exact wording.

## Reproduce

Mutmut is a local development tool and is not part of the lock files. Run it on a
temporary copy so its configuration does not enter the repository:

```bash
cp -R src tests test_data pyproject.toml /tmp/mutation/
cat >> /tmp/mutation/pyproject.toml <<'EOF'
[tool.mutmut]
source_paths = ["src"]
only_mutate = ["src/llm_testkit/evaluation/correctness.py"]
also_copy = ["test_data"]
pytest_add_cli_args_test_selection = ["tests/unit/test_correctness.py"]
EOF
cd /tmp/mutation && mutmut run && mutmut results
```

Use `source_paths = ["src"]`, not only the mutated file: otherwise the mutated
package lacks its `__init__.py` files and the editable install of the original
package is imported instead.

## Framework findings deferred to a later step

- `validate_control`, `validate_result` and `evaluate_correctness_report` use
  `assertions.assert_quality_score` and `assert_model_available`. These raise
  `AssertionError` through `assert` statements, which Python removes under `-O`,
  and callers must handle two exception types for one validation contract.
- `evaluate_correctness_report` re-checks that a control response is a nonempty
  string after `validate_control` has already done so. The branch is unreachable.

## Migration log

One domain per commit. Mutation scores use the same production modules before and
after; "before" runs the old test module, "after" the rewritten one(s).

| Domain | Production modules | Test modules | Mutants detected before → after |
| --- | --- | --- | ---: |
| Correctness | `evaluation/correctness.py` | `test_correctness.py` | 49.7% → 92.2% |
| Faithfulness and judge | `evaluation/faithfulness.py`, `evaluation/ollama_judge.py` | `test_faithfulness.py`, `test_ollama_judge.py` | 58.6% → 94.1% |
| Context relevance | `evaluation/relevance.py` | `test_relevance.py` | 49.7% → 94.0% |
| Faithfulness controls | `evaluation/calibration.py`, `core/number_words.py` | `test_calibration.py` | 34.2% → 92.0% |

Domains not listed still follow step 35.

Findings recorded during migration, for the framework step:

- Faithfulness: an empty verdict list becomes a NaN score and is rejected as an invalid
  quality score (`AssertionError`) rather than as missing verdicts (`ValueError`).
- `assert_quality_score(minimum=...)` is used only by tests; no gate calls it.
- Calibration separates `matched` from `mismatch` by catching the `AssertionError` raised by
  `assert_calibration_result`. Under `python -O` those assertions are removed, so every
  control would be reported as matched regardless of the judge's verdicts.
- CLI tests that omit a required argument must run in a temporary directory. Otherwise a
  mutant that accepts the missing argument writes `out.json` to the working directory,
  and a later run fails for an unrelated reason, so the mutant survives undetected.
