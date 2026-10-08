# Judge suitability checks and disagreement review

## Purpose and decision

This stage investigates the failed first benchmark before using semantic scores
as release criteria. It adds a source-bound control catalog for all four metrics,
a bounded real-judge runner, and an offline disagreement report. Existing RAGAS
algorithms, golden references, application answers and score thresholds are unchanged.

The current decision is **release suitability not established**. Retain the
experimental thresholds for explicit demonstrations; do not describe them as
calibrated release acceptance criteria. A small control check is a prerequisite
for investigating a judge, not sufficient evidence to approve one.

## Review of the original experiment

The preserved four-case Qwen3.5 4B benchmark on 2026-10-07 failed, although all
four answers passed the reviewed fact/source checks and resource cleanup. The
offline review revalidates the original benchmark, samples, JUnit, metric evidence
and checksums before extracting disagreement candidates. Rule checks are limited:
their pass does not establish that every assertion in an answer is correct.

| Case | Observation | Engineering assessment |
| --- | --- | --- |
| `paid_leave` | All applicable checks passed | No disagreement identified in this sample |
| `leave_approver` | Correctness 0.0; faithfulness 1.0 | Correctness evaluated the answer against a short reference, not the policy or question. Source attribution was absent from that reference; claim extraction added an approval-authority paraphrase; verification lost the paid-leave meaning of "the request". Investigate metric input scope and decomposition, rather than treating 0.0 as proof of a wrong approval role. |
| `hotel_receipt_condition` | Correctness 0.0; faithfulness 0.5 | The answer included the documented KGS 8400 limit, absent from the short reference. Reference-only scoring cannot establish whether this extra fact is supported by the policy. Faithfulness also rejected the inference from "with a receipt" to a receipt requirement. The golden contract interprets this as a requirement; that interpretation needs explicit review. |
| `gym_missing` | Refusal rules and sources passed | Semantic metrics are N/A under the existing refusal contract; no score is invented |

These are engineering assessments, not independent human adjudication. No reviewer
identity, approval or signature is fabricated. The original `human-review.json`
and follow-up worksheet remain pending and unchanged. The failed experiment is
retained; this stage does not replace it with a passing rerun.

## Control design

`automation/test_data/judge-validation.json` contains engineering-authored labels,
their rationale, and exact hashes of the golden dataset and policy. Every question
and reference must match a golden case. Synthetic contexts must be exact excerpts
of the bound policy. These are metric controls, not real retrieval observations or
new application-generated evidence.

| Metric | Positive/control contrast | Expected evidence |
| --- | --- | --- |
| Faithfulness | Receipt required / reimbursement without a receipt | Supported / contradicted claim, scores 1 / 0 |
| Factual correctness | Manager approval with policy attribution / finance approval | Accepted task meaning / wrong approval role, expected scores 1 / 0 |
| Context precision | Useful leave excerpt first / second, with the same unrelated excerpt | Ordered labels `[1, 0]` / `[0, 1]`, average precision 1 / 0.5 |
| Context recall | Entitlement and notice retrieved / entitlement only | Both facts supported / notice absent, recall 1 / 0.5 |

Correctness labels express the intended policy-task acceptance contract. A
disagreement can reveal a reference-scope limitation as well as a judge limitation;
it is not automatically an incorrect implementation of factual F1. Receipt labels
use the existing golden interpretation of the documented condition.

The runner compares both scores and claim/context verdicts with labels. Reversed
fact verdicts cannot pass merely because their average is unchanged. Unexpected
claims require review. Raw verdicts independently reproduce the reported math;
malformed or incomplete evidence is an error, distinct from a label mismatch.

## Run and inspect

Run from `automation` with the existing evaluation dependencies. Preview makes no
network calls, and validates catalog provenance, labels, selection and budget:

```bash
.venv/bin/python -m llm_testkit.evaluation.judge_validation --dry-run
```

The default experiment permits **18 judge calls**, runs serially, generates no
application answers, and does not retry. It requires native Ollama with the chosen
model already installed; no application restart or capture overlay is required.

```bash
.venv/bin/python -m llm_testkit.evaluation.judge_validation \
  --output reports/judge-validation/run-001.json
```

To isolate one diagnostic without running the full catalog:

```bash
.venv/bin/python -m llm_testkit.evaluation.judge_validation \
  --control manager_policy_paraphrase --max-judge-calls 4 \
  --output reports/judge-validation/manager-001.json
```

Exit 0 means all selected labels matched, 1 means mismatches or execution errors,
and 2 means invalid arguments/data/budget. Existing output files are refused before
network access. Raw prompts, structured outputs, model digest, RAGAS version,
judge settings, inputs, labels and provenance hashes are retained. Missing model
or transport errors are saved rather than reported as matching controls.

Extract diagnostics from an existing benchmark with **zero model calls**:

```bash
.venv/bin/python -m llm_testkit.reporting.judge_review \
  reports/benchmark-2026-10-07-smoke/benchmark.json \
  --output reports/judge-validation/original-benchmark-review.json
```

This command exits 0 for a successfully built review, even if the original benchmark
failed. Its `original_benchmark_status` preserves that outcome. It never approves,
rescales, edits or reclassifies original evidence. Reports stay ignored by Git.

## Observed judge check on 2026-10-08

The full eight-control experiment used Qwen3.5 4B, digest
`2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`,
RAGAS 0.4.3 and the existing judge settings (temperature 0, seed 42, thinking
disabled). All 18 calls completed: **7 matched, 1 mismatched, 0 errors**.

`manager_policy_paraphrase` scored **0.67**, rather than the expected 1.0. The
judge accepted the manager-approval fact, but extracted a second claim that the
approval follows policy and rejected it because the short reference did not say
that. Both reference claims were supported. This isolates reference sensitivity
without a random file name or ambiguous "the request" in the answer. Both receipt
controls matched in this run; this does not erase the earlier context-dependent
faithfulness disagreement.

The experiment returned nonzero and was retained at
`automation/reports/judge-validation-stage-1-live.json`, SHA-256
`2c0b8f18f744737eee790965c6405ac91f5fb87d3638c43167dbeae913a56d97`.
The original benchmark SHA-256 is
`1a5dae262a85ef01713f27dedbfd993bbd76671ba1c80ccb8c5e2c74f2b3ea1c`.
Seven matches are not a population accuracy estimate or calibration of thresholds.

## Acceptance criteria before promotion

1. Review policy-task labels and disputed answers against the actual source and
   question, recording who reviewed them without changing the historical run.
2. Decide whether correctness is intended to measure only reference overlap or
   all source-supported answer claims. Do not strip citations, broaden references
   from a failing answer, or relax a threshold solely to obtain a pass.
3. Version any justified reference, prompt or judge change and assess it on
   separate examples, including wrong roles, wrong numbers, unsupported claims,
   omitted facts and irrelevant contexts. Preserve old results for comparison.
4. Establish acceptable judge disagreement and threshold criteria on a broader
   labelled evaluation set. Matching these eight controls alone is insufficient.

This stage completes the reproducible diagnosis and suitability-check mechanism;
independent human approval and statistical calibration are not claimed.

## Framework layers and verification

- `datasets/judge_controls.py`: catalog validation, source hashes and call planning.
- `evaluation/judge_validation.py`: existing RAGAS scorers, raw-evidence math and label comparison.
- `reporting/judge_review.py`: verified saved-benchmark diagnostics without inference.
- `test_support/{data,builders,fixtures,assertions}`: independent unit data and readable scenarios.

Unit checks exercise real RAGAS metrics with mocked judge responses, stale source
rejection, score/verdict inconsistencies, mismatches and errors that preserve later
checks, preview/overwrite safety, and review integrity. They make no model calls.

The final local framework check passed **557 unit cases** with **81.68%** combined
statement/branch coverage, above the configured 75% floor. Ruff, isort and YAPF
checks passed. The real eight-control experiment is recorded separately above;
its mismatch is preserved and is not counted as a passing application test.
