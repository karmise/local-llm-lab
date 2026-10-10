# Step 40: framework fixes found by the test migration

[Step 39](step-39-test-architecture.md) recorded framework defects that the rewritten
tests exposed. This step fixes them, one commit each, with a test that fails on the
previous code.

## Part 1: correctness fixes

| Fix | Previous behaviour | Regression test |
| --- | --- | --- |
| Judge controls no longer depend on `assert` | `evaluate_controls` told `matched` from `mismatch` by catching `AssertionError`; under `python -O` every control was `matched`. Saved benchmarks were checked the same way. | `test_mismatch_is_detected_under_optimized_python` runs a contradicting judge under `-O` |
