# Counterfactual bias acceptance checks

Three reviewed pairs change only an employee descriptor: gender, age or nationality. Each pair asks the same carryover question, uses the same policy, and shares the original golden reference, facts and sources. The fictional policy does not define different carryover entitlements for these descriptors. The catalog is source-bound and validates that no other question text changes.

Each variant gets an independent temporary workspace. JUnit and capture metadata retain pair/variant IDs and catalog hashes. Shared assertions reject labelled unsupported eligibility restrictions and stereotyped capability claims. Stability reports keep variants separate. Passing both variants means both satisfy the same acceptance criteria, rather than requiring identical wording.

From `automation`:

```bash
.venv/bin/python -m pytest tests/test_bias.py --run-bias --rag-model qwen3.5:4b \
  -k gender_carryover --junitxml reports/bias/runs.xml
.venv/bin/python -m llm_testkit.reporting.bias reports/bias/runs.xml \
  --pair gender_carryover --model qwen3.5:4b --output reports/bias/comparison.json
```

This selection makes two generation calls. A full default run makes 12: three pairs × two variants × two models. Repeat pair/model flags to declare a larger comparison scope; repetitions must match `--repeat`. Tests skip unless explicitly enabled.

One acceptance failure and one pass is reported as **asymmetry**. Two failures are **shared failure**, not evidence of equal acceptable treatment. Missing/skipped/error runs, duplicate observations, stale catalogs or mismatched model/prompt/retrieval settings make the result incomplete. The comparison exits nonzero unless all declared pairs pass.

This is a small counterfactual regression suite. It does not establish demographic fairness, clinical bias, equivalent tone, or the absence of stereotypes beyond the reviewed lexical criteria. It does not use invented demographic outcome statistics. Broader validation needs representative data, domain-reviewed criteria and enough independent observations. Captured modified questions cannot be relabelled as ordinary golden samples for semantic evaluation: those evaluators deliberately require an exact question/reference binding.
