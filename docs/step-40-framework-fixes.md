# Step 40: framework fixes found by the test migration

[Step 39](step-39-test-architecture.md) recorded framework defects that the rewritten
tests exposed. This step fixes them, one commit each, with a test that fails on the
previous code.

## Part 1: correctness fixes

| Fix | Previous behaviour | Regression test |
| --- | --- | --- |
| Judge controls no longer depend on `assert` | `evaluate_controls` told `matched` from `mismatch` by catching `AssertionError`; under `python -O` every control was `matched`. Saved benchmarks were checked the same way. | `test_mismatch_is_detected_under_optimized_python` runs a contradicting judge under `-O` |
| Faithfulness validation order and error type | An empty verdict list reached the score check first and failed as an invalid score (`AssertionError`, skipped under `-O`) instead of as missing verdicts. | `test_scoring_rejects_invalid_judge_verdicts[no-verdicts]`, `test_validate_result_rejects_invalid_score` |
| Score validation raises `ValueError` everywhere | Correctness F1 bounds and values, judge-control scores and faithfulness scores were checked with an `assert`-based helper, so `python -O` accepted NaN or out-of-range scores, and callers caught two exception types. `core/scores.require_score` replaces it. The unreachable second "control response must be nonempty" check in `evaluate_correctness_report` is removed. | New `nan-bound`, `bound-above-one`, `nan-score` cases in `test_correctness.py`; `invalid-score` in `test_judge_validation.py` |
| Capture marker must be the exact end of the prompt | The pattern ended in `$`, which also matches before a final newline, so `"Policy\n<marker>\n"` normalised to `"Policy\n"`. It now ends in `\Z`. | `test_other_prompt_text_is_kept[before-final-newline]` |
| Drift requires the captured policy and reports a prompt change once | `make_snapshot` read `metadata["policy_sha256"]` directly, so a sample without it failed with `KeyError`; a mismatching value was already rejected by `bind_case`. A changed prompt appeared as both `prompt_sha256` and `retrieval_configuration`. | `test_make_snapshot_requires_captured_policy`; `test_comparison_reports_changed_conditions[prompt]` |

## Part 2: structural changes

| Change | Why |
| --- | --- |
| `assert_quality_score` loses its unused `minimum` argument | No gate passed it; thresholds belong to `reporting/gates.py`. |
| `assertions.py` (480 lines) becomes the `assertions/` package: `fields`, `api`, `answers`, `quality`, `ui` | Each area can be read and changed alone; `assertions.assert_...` calls are unchanged because the package re-exports every check. |
| `options.py` hooks become one function per rule, and catalog errors are usage errors | The two hooks held every selection rule in about 120 lines. An invalid catalog or unknown prompt variant raised `UsageError` inside `pytest_generate_tests`, which pytest reports as a collection error (exit code 2). It is now recorded and raised after collection as a usage error (exit code 4) with the same message. |

Not done: typed catalog loaders (for example pydantic). The hand-written validators are covered
by tests that detect 90–96% of mutants, and a new runtime dependency would change the reviewed
lock files used for IQ. It is a separate decision, not a cleanup.

## Found by the live gate

| Fix | Previous behaviour | Regression test |
| --- | --- | --- |
| Number words above 99 are normalised | The live judge rewrote "KGS 5000" as "five thousand KGS". `normalize_number_words` converted only 0–99, so the labelled claim `\b5000\b` matched nothing and a correct judge was reported as a control mismatch, failing the benchmark. Well-formed integers of any size are now written as digits; decimals and malformed phrases are kept. | `test_number_spellings[judge-rewritten-amount]` and new size and malformed cases; both saved CI calibrations now match |
