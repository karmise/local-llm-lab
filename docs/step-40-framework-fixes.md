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

## Part 2: structural changes (next)

- Split `assertions.py` (API, answer, quality and UI checks in one 480-line module).
- Reduce `pytest_support/options.py` complexity: one selection rule per marker instead of
  one long hook, and raise collection-time selection errors as usage errors.
- Typed catalog loaders (for example pydantic models) instead of hand-written field checks.
- Remove `assert_quality_score(minimum=...)`, which no gate uses.
