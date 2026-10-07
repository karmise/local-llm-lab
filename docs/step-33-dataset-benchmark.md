# Step 33: Dataset quality benchmark

## Purpose and architecture

The benchmark connects existing golden application tests and the three RAGAS
evaluation services into one bounded experiment. It measures faithfulness,
factual correctness, context precision and context recall for every applicable
case, then reports outcomes by generation model and question category.

- `datasets/benchmark.py` selects reviewed golden cases and calculates the budget.
- `evaluation/benchmark_runner.py` runs one existing golden pytest case at a time.
  Existing API clients and resource fixtures own setup, indexing and verified
  teardown. Setup, assertion and teardown failures remain in JUnit evidence.
- `evaluation/benchmark.py` validates captured sample provenance, applies shared
  answer/source assertions, calls existing evaluators and validates original
  judge evidence independently for each metric.
- `reporting/benchmark.py` aggregates the declared matrix, including missing
  cases and errors. `benchmark_evidence.py` revalidates saved artifacts offline.
- `tests/test_benchmark_report.py` optionally exposes the aggregate in Allure.

The runner uses the document-only Query profile and temporary workspaces. It does
not change the conversational profile or documents in the manual workspace.
Actual model contexts come from the existing observation overlay, not from the
sources displayed alongside a response. Failed checks are never retried until
they pass. A failed answer can still be evaluated, preserving independent
evidence about why it failed; a missing captured answer becomes an error.

## Default scope and budget

The default model is `qwen3.5:4b`, with four explicitly curated cases:

| Case | Category | Purpose | Semantic metrics |
| --- | --- | --- | --- |
| `paid_leave` | Multi-fact | Entitlement, notice period and day units | All four |
| `leave_approver` | Fact lookup | Correct approval role | All four |
| `hotel_receipt_condition` | Boundary | Resist a misleading receipt premise | All four |
| `gym_missing` | Missing information | Acknowledge absence without inventing compensation | N/A |

Refusal cases still require reviewed answer rules, forbidden-content checks and
the expected document sources. In this benchmark, semantic claim/retrieval
metrics are explicitly outside the refusal scope: N/A is not zero, one or a
judge failure. Refusal acceptance is reported separately by category. Other
projects can define a dedicated abstention metric and reviewed acceptance data.

Three hand-labelled faithfulness controls run once using the first captured
paid-leave context: a supported paraphrase, wrong numbers and an unsupported gym
amount. Each uses at most two judge calls. This requires selecting `paid_leave`.
Control mismatch or error prevents the aggregate from passing, while the other
case evidence remains available for diagnosis.

Execution is serial, with one generation per selected case/model and no retries.
Each positive case needs at most 12 total model calls: one generation, two
faithfulness calls, four correctness calls and up to five relevance calls. A
refusal needs one generation. The three controls add six calls per run. The
default four-case run therefore allows at most **43** calls; the same cases on
two models allow at most **80**. Ordinary unit/API runs do not invoke the runner.
The budget counts generation and judge calls; document embedding during fixture
setup is separate. The observed first run used four generations and 33 judge calls.
Requests retain configured timeouts. Generation thinking uses the model default;
judge thinking is disabled. No automatic outer process kill bypasses pytest's
resource teardown.

## Run and inspect

Use the existing Python 3.12 environment and pinned evaluation dependencies.
From `automation`, preview the default plan without network or model calls:

```bash
.venv/bin/python -m llm_testkit.evaluation.benchmark_runner \
  --output reports/benchmark-preview --dry-run
```

Enable actual context observation from the repository root:

```bash
docker compose -f compose.yaml -f compose.capture.yaml up -d anythingllm
```

Wait for application readiness, then run from `automation`:

```bash
.venv/bin/python -m llm_testkit.evaluation.benchmark_runner \
  --output reports/benchmark-first
```

Use a new directory for each experiment. Existing outputs are rejected rather
than overwritten. An explicit two-model comparison is:

```bash
.venv/bin/python -m llm_testkit.evaluation.benchmark_runner \
  --model qwen3.5:4b --model qwen2.5:7b \
  --max-model-calls 80 --output reports/benchmark-two-models
```

Repeat `--case` to select other golden cases. Keep `paid_leave` for judge controls.
The complete 16-case dataset on two models needs at most 324 calls; select all
case IDs and a sufficient explicit budget. There are no hidden model repetitions.
Two generation models are a hard limit for this local runner.

The output contains:

- `manifest.json`: dataset/policy/control hashes, thresholds, planned matrix and budget.
- Exact copies of the golden dataset, source policy, thresholds and control catalog.
- `case-NNN/`: generated sample, pytest log/JUnit, raw judge reports and case result.
- `judge-controls.json`: labelled controls, extracted claims, verdicts and mismatches.
- `benchmark.json` and `benchmark.md`: per-model/category results and case diagnostics.
- `human-review.json`: answer/reference worksheet with pending reviewer decisions.

Metric means use available measurements and always show measured/eligible,
unavailable, below-threshold and N/A counts. Overall pass rate uses the **planned**
number of case/model pairs. One failed case prevents acceptance even if averages
are high. Missing results, changed weights/settings or judge/control errors
prevent a passing aggregate. Models must share workspace settings apart from
their generation model and unique capture marker.

The CLI exits zero only for `checks_passed`, one for failed/error experiments and
two for invalid arguments/budgets. Retain failed runs as evidence. Compare runs
only when dataset, prompt, model digests, judge and settings are controlled.

For offline Allure reporting, without further generation or judge calls:

```bash
.venv/bin/python -m pytest tests/test_benchmark_report.py \
  --benchmark-report reports/benchmark-first/benchmark.json \
  --alluredir reports/benchmark-first/allure-results \
  --allure-no-capture --junitxml reports/benchmark-first/benchmark-results.xml
```

Saved reporting checks copied baselines, case/sample/JUnit checksums, raw metric
evidence and hand-labelled controls before recomputing the aggregate. It does
not trust an edited top-level pass flag. Reports are local and ignored by Git.
Keep all case artifacts beside the aggregate when moving a report.

After a live run, restore ordinary infrastructure from the repository root:

```bash
docker compose -f compose.yaml up -d anythingllm
```

## Acceptance rationale and review limits

Reviewed golden facts, units, source fragments and forbidden claims remain the
primary acceptance criteria. The semantic thresholds reuse the explicitly
experimental gate catalog: faithfulness 0.9, correctness 0.8, context precision
0.8 and context recall 0.9. They are illustrative floors, not numbers derived
from this small benchmark or endorsed by an independent expert.

The controls check a small part of judge behaviour. They do not calibrate every
metric, establish inter-rater agreement or make a judge independent of the
generation model. A report records the selected judge identity; using the same
model family can introduce correlated errors. Review `human-review.json`
against the captured contexts, record disagreement examples and preserve a
separate reviewed copy before proposing threshold changes.

## First observed run

On 2026-10-07 the four default cases ran on Qwen3.5 4B. All four passed the
reviewed answer/source checks and resource cleanup. All three judge controls
matched their labels. The aggregate nevertheless failed: the manager-approval
answer scored 0.0 for correctness, and the receipt-condition answer scored 0.0
for correctness and 0.5 for faithfulness. Precision and recall passed on all
three applicable cases. No errors or missing cases were hidden in the aggregate.

The preserved verdicts show why review is necessary. The judge penalized policy
file attribution absent from a short reference, treated "the request" as less
specific than "paid-leave request", and refused to infer a receipt requirement
from "with a receipt". A narrow reference and an overly literal judge can both
affect scores. These are disagreement candidates, not automatically proven
application defects. The failed run was retained without changing thresholds or
repeating generations to obtain a passing result. Only this one-model subset
was executed; two-model orchestration was verified with offline tests and a dry run.

Four curated questions provide reproducible examples and diagnostics, not a
population accuracy estimate, confidence interval or clinical validation.
The runner enables larger explicit selections without claiming that a complete
two-model matrix has already been executed. Automated fresh AI checks in CI and
enforced branch protection remain separate work.
