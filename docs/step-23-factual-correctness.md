# Step 23: Reference-based factual correctness

## What changed

The evaluation layer now supports RAGAS `FactualCorrectness` alongside
faithfulness. It compares a captured answer with the reviewed reference from the
golden catalog, rather than with retrieved context. F1 accounts for unsupported
response claims and omitted reference claims. It is an exploratory measurement,
not a calibrated acceptance threshold or an estimate of general model accuracy.

The implementation uses the existing pinned RAGAS 0.4.3 and native Ollama
adapter. No cloud API or additional embedding service is introduced. This is the
`FactualCorrectness` metric, not RAGAS's separate `AnswerCorrectness` metric that
combines factual and similarity components. See the
[official metric documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/factual_correctness/).

Configuration is explicit: F1 mode, beta 1, high atomicity and high coverage.
A measurement makes at most four sequential judge calls: decompose the answer,
verify those claims against the reference, decompose the reference, and verify
those claims against the answer. The existing faithfulness budget remains two;
existing live-quality behavior remains one generation plus two judge calls.
Correctness does not run automatically during golden or live-quality tests.

## Evidence and validation

[correctness.py](../automation/src/llm_testkit/evaluation/correctness.py) provides
scoring, report validation, synthetic control checks and an explicit CLI.

Reports retain both sets of claims, every verdict/reason, response-side TP/FP,
reference-side FN, F1, raw judge calls, model digest, metric/judge configuration,
response origin, and sample/dataset/reference checksums. The pinned implementation
uses response-side TP and reference-side FN for its F1 and rounds to two decimals;
the validator recomputes that score from verdicts. These counts are outputs of a
judge and are not independent human labels.

Sample loading checks captured context integrity. Before a judge call, the
question and reference must match the selected golden case, and any recorded
policy/dataset metadata must match the current catalog. Completed evidence must
belong to the same sample and dataset, retain the same answer/reference and metric
configuration, and agree with its raw call outputs. Incomplete coverage, duplicate
claims, invalid verdicts, inconsistent scores and failed/truncated generation are
errors; there are no retries. The transport is closed on success and failure.
Checksums bind files but do not prove the truth of labels or provide authenticity
against someone who rewrites the entire evidence package.

## Small hand-labelled controls

[correctness-controls.json](../automation/test_data/correctness-controls.json)
contains a correct answer, paraphrase, incomplete answer and contradicted answer
for the paid-leave case. Controls require expected response verdicts, labelled
reference facts and an expected F1 range. Every reference claim must be covered
by a label. Labels never enter judge prompts.

The incomplete control expects a partial score and specifically requires the
notice-period claim to be judged absent. A fixed expected score would assume a
fixed claim count: splitting one supported fact into several claims changes F1.
This is why the control validates the omission itself as well as the score range.
These four examples are diagnostics, not statistical judge calibration. Synthetic
control reports cannot be supplied as application-quality evidence.

## Explicit execution

From `automation`, install `requirements-evaluation.lock` and use a saved sample
whose question/reference matches the golden catalog. Run one control:

```bash
python -m llm_testkit.evaluation.correctness reports/rag-samples/<capture-id>.json \
  --case paid_leave --control incomplete \
  --output reports/correctness-controls/incomplete.json
```

Evaluate the saved application answer separately:

```bash
python -m llm_testkit.evaluation.correctness reports/rag-samples/<capture-id>.json \
  --case paid_leave --judge-model qwen3.5:4b \
  --output reports/correctness/application.json
```

Each command permits at most four judge calls, with thinking disabled and no
retries. It does not generate a new application answer. A control mismatch or
evaluation error returns a nonzero exit code and retains the evidence. An
existing output file is never overwritten. Use `--dataset` and `--policy` for
explicit alternate paths. Other golden cases can be evaluated with their case ID
and corresponding captured question/reference.

## Optional fourth dimension in the combined Allure report

With reporting dependencies installed, add existing correctness evidence to the
saved paid-leave quality scenario:

```bash
python -m pytest tests/test_quality.py \
  --quality-sample reports/rag-samples/<capture-id>.json \
  --faithfulness-report reports/faithfulness.json \
  --correctness-report reports/correctness/application.json \
  --alluredir reports/quality-correctness/allure-results
```

Both judge files must belong to the exact same captured sample. This command
makes no model calls. Existing runs without `--correctness-report` retain the
three-dimensional report. With it, the report shows required facts, sources,
faithfulness and factual correctness independently. The saved pytest scenario
currently targets paid leave; the evaluation CLI supports other golden case IDs.

A valid low correctness score is recorded as `measured`, not a threshold failure.
Required-fact/source failures remain failures; malformed correctness evidence is
an independent error. All dimensions are rendered even when one fails. This
stage does not implement context precision/recall or a release quality gate.

## Verified results

On 2026-10-06, all 204 unit checks passed. Real RAGAS executed with mocked judge
responses for correct, incomplete, contradicted and extra-claim cases; validation
also covered sample/reference binding, malformed evidence, control expectations
and transport cleanup. Ruff passed.

The local Qwen3.5 4B judge evaluated one synthetic incomplete answer and one
previously captured application answer, using eight calls in total and no new
application generations. The incomplete answer scored 0.8 because the judge
extracted two supported response claims and one missing reference claim. Its
initial fixed-score expectation of 0.67 mismatched. That original report was
preserved; the same raw output was then checked against labelled omission rules,
without another model call. The notice-period omission was correctly identified.

The application answer scored 1.0. A saved Allure run passed with all four
dimensions and generated a report. Other synthetic controls were verified through
offline protocol tests; they were not all run against the real judge. Local raw
reports remain ignored under `automation/reports/correctness-stage/`.
