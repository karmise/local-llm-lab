# Real two-model quality comparison

## Scope

This stage executes the existing source-bound benchmark on both installed
generation models, Qwen3.5 4B and Qwen2.5 7B. It compares four curated questions
per model, with one answer per question, serial execution and no retries.

| Case | Category | Acceptance focus |
| --- | --- | --- |
| `paid_leave` | Multi-fact | 23 working days, 12 calendar days' notice, correct units |
| `leave_approver` | Fact lookup | Direct manager approval |
| `hotel_receipt_condition` | Boundary | Receipt condition despite a misleading premise |
| `gym_missing` | Missing information | Explicit absence of a policy without an invented amount |

Each case uses a fresh temporary indexed workspace and verified resource cleanup.
The manually used conversational workspace is not switched to a benchmark model.
The source policy, golden dataset, prompt, temperature (0.1), history (0), topN (4),
similarity threshold (0.25), assertions and experimental semantic thresholds are
shared. The report rejects changed configuration or generation weights.

Qwen3.5 4B is the fixed judge for both models. Judge temperature is 0, seed 42,
thinking disabled, and retries disabled. Actual context is captured at the Ollama
SDK boundary; citations alone are not treated as the context seen by the model.

The first-stage [judge suitability check](step-36-judge-validation.md) found a
reference-attribution mismatch. That finding remains applicable. The benchmark's
three faithfulness controls are a separate narrow sanity check, not approval of
all four metrics. Using the same model for one generator and the judge also
introduces correlated-error and possible self-preference risks.

## Timing added to the shared framework

`rag_chat` records monotonic elapsed time around the completed AnythingLLM chat
API call. `answer_request_seconds` is written to JUnit properties and captured
sample metadata, then carried into benchmark rows. Fresh and saved-report
validation bind row timing to the sample and JUnit evidence.

This duration includes request handling, retrieval, model loading, generation and
receipt of the response body. It excludes document setup/indexing, judge calls,
assertion/report processing and resource cleanup. It is not time to first token
or isolated model inference time. Failed semantic checks still retain timings
for completed captured answers; missing measurements remain unavailable.

Reports show measured/planned counts and mean/minimum/maximum durations per model.
Older captured samples remain readable and show unavailable timing, never zero
or a duration derived from fixture execution. Invalid, non-finite or non-positive
durations are rejected.

Model loading and cache state are not controlled. The models differ in size and
default reasoning behavior; no generation seed or identical compute budget is
claimed. A single observation per question is descriptive evidence, not a latency
SLO, stability measurement or statistically reliable speed ranking.

## Reproduce

Prerequisites: the existing Python 3.12 environment with evaluation dependencies,
native Ollama with both generation models and the embedding model installed,
AnythingLLM online, and the existing local developer API key configured.

From the repository root, temporarily enable actual context observation:

```bash
docker compose -f compose.yaml -f compose.capture.yaml up -d anythingllm
```

From `automation`, preview the plan without application/model calls:

```bash
.venv/bin/python -m llm_testkit.evaluation.benchmark_runner \
  --model qwen3.5:4b --model qwen2.5:7b --max-model-calls 80 \
  --output reports/benchmark-two-models-next --dry-run
```

After application readiness, run the same command without `--dry-run`. Choose a
new output directory for each experiment. The budget allows eight generations
and at most 72 judge calls, including six calls for the three faithfulness controls.
Embedding during fixture setup is outside that model-call budget. A failed run is
retained and is not retried to obtain a passing report.

From the repository root, restore ordinary infrastructure after the experiment,
including after failures:

```bash
docker compose -f compose.yaml up -d anythingllm
```

For saved Allure reporting from `automation`, without new model calls:

```bash
.venv/bin/python -m pytest tests/test_benchmark_report.py \
  --benchmark-report reports/benchmark-two-models-next/benchmark.json \
  --alluredir reports/benchmark-two-models-next/allure-results \
  --allure-no-capture --junitxml reports/benchmark-two-models-next/benchmark-results.xml
```

The saved-report test preserves the aggregate's acceptance outcome. Nonzero exit
for failed quality criteria does not mean report generation failed. A high metric
mean cannot override a failed case, missing evidence or an invalid judge result.

The output directory retains the input baselines, eight case directories, raw
judge evidence, captured samples, generation JUnit/logs, aggregate JSON/Markdown
and a pending human-review worksheet. These local artifacts remain ignored by
Git; the checked-in results below describe this specific experiment.

## Observed results on 2026-10-08

All eight planned case/model pairs completed, including verified fixture cleanup.
There were **zero execution errors, zero missing results and zero retries**.
Both models passed all four reviewed fact/source checks. The overall benchmark
nevertheless returned **failed**, preserving three below-threshold cases.

| Generation model | Fact/source checks | Cases passing all experimental criteria | Mean faithfulness | Mean correctness | Mean precision / recall | Mean answer seconds | Min / max seconds |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| Qwen3.5 4B | 4 / 4 | 2 / 4 | 0.917 | 0.480 | 1.000 / 1.000 | 54.936 | 22.968 / 111.974 |
| Qwen2.5 7B | 4 / 4 | 3 / 4 | 1.000 | 0.890 | 1.000 / 1.000 | 6.865 | 5.968 / 8.641 |

Metric means cover three applicable questions per model. The gym refusal uses
reviewed rules and sources; all four semantic metrics are N/A, not perfect scores.
Timings cover all four completed answers per model.

| Case | Qwen3.5 4B | Qwen2.5 7B |
| --- | --- | --- |
| `paid_leave` | Passed; faithfulness/correctness 1.0 / 1.0; 31.991 s | Passed; 1.0 / 1.0; 6.424 s |
| `leave_approver` | Failed correctness; 1.0 / 0.0; 22.968 s | Passed; 1.0 / 1.0; 5.968 s |
| `hotel_receipt_condition` | Failed faithfulness and correctness; 0.75 / 0.44; 111.974 s | Failed correctness; 1.0 / 0.67; 8.641 s |
| `gym_missing` | Passed refusal rules/sources; metrics N/A; 52.812 s | Passed refusal rules/sources; metrics N/A; 6.429 s |

The experiment used **68 generation/judge calls**: eight application generations,
54 per-case judge calls and six faithfulness-control calls, below the declared
80-call ceiling. All three narrow controls matched their labels. Every positive
sample contained two actual context blocks. Fresh and saved-report checks agreed
on the complete matrix and its failed aggregate.

### Failure interpretation

- Qwen3.5's manager answer included a valid policy-file attribution and used
  "the request". The correctness judge rejected source attribution absent from
  the short reference and gave 0.0; faithfulness against actual context was 1.0.
  Qwen2.5's short manager answer matched the reference and scored 1.0.
- Qwen3.5's hotel answer included the documented amount, company/version details,
  and cautious language about claiming without a receipt. Some extra facts were
  absent from the short reference. Faithfulness also rejected the cautious claim
  about what the document explicitly states. Review both the judge's entailment
  and the answer's handling of the intended receipt requirement.
- Qwen2.5's hotel answer stated the receipt requirement and scored faithfulness
  1.0. Correctness scored 0.67 because the judge rejected the source filename and
  KGS 8400 limit absent from the short reference, while accepting the receipt facts.

Passing regex/source rules does not prove all answer claims correct. Conversely,
a reference-overlap failure does not prove that an extra claim contradicts the
policy. The raw verdicts preserve these distinctions for review. Do not infer
overall model superiority from this sample or label a single model difference
as flakiness. The observed latency difference applies to these requests and the
models' default reasoning/load conditions.

### Evidence and reproducibility

Local experiment directory: `automation/reports/benchmark-two-models-2026-10-08`.
Its `benchmark.json`, `benchmark.md`, `human-review.json`, raw case artifacts,
Allure results and generated HTML report are retained. Saved Allure rendering
correctly produces one failed aggregate test, with all eight case steps retained.
That rendering's test duration measures offline report validation, not the earlier
model requests; use the explicit answer-timing table and attachments for those.

- Generation/judge Qwen3.5 digest:
  `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`.
- Generation Qwen2.5 digest:
  `845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e`.
- Golden dataset SHA-256:
  `60b8edba5525ee74a2ed28472c2d9d6fcf192b98909b6295b84bd72b0a12fcf6`.
- Source policy SHA-256:
  `02ca83ffa0e66349e39eeb58de78fd1010471f2dd892e6c9739346000e841709`.
- Framework source SHA-256, identical in all eight captures:
  `57f95af3f78746d66ba7fb3bc6eb9a2dcd690a0ef53f873a260ad1c2b54432b8`.
- Benchmark JSON SHA-256:
  `618b89fe48056cc10b493ebb94318c3615d713b0a407357071051fdc8bdd97de`.

The orchestration restored ordinary Compose configuration in a `finally` block.
After restoration, application health returned 200 and `company-policy-lab`
remained in Chat mode on Qwen3.5 4B.

No references, thresholds or application settings were changed to obtain a pass.
This failed result is not promoted to an approved baseline. Human review and
release suitability remain pending; future comparisons must preserve equivalent
conditions and record any justified changes explicitly.

## Framework verification

The offline suite passed **566 unit cases** with **81.84%** combined
statement/branch coverage, above the 75% floor. Timing checks cover measurement
boundaries, descriptive aggregation, rejection of invalid durations and backward
compatibility with older samples that have no timing data. The real saved report
also revalidated duration equality across all sample, row and JUnit artifacts.

Ruff, isort and YAPF checks passed. The saved quality-report test deliberately
remains failed because the actual benchmark did not meet its experimental gates;
that outcome is distinct from the passing framework unit suite.
