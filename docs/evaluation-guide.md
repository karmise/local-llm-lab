# Evaluation guide: opt-in suites and evidence tools

This is the command reference for everything beyond the default unit, API and RAG
layers. Each section says what the suite measures, how much model work it costs
and how to run the smallest useful scope. Design reasons are in
[decisions.md](decisions.md); the layer overview is in the
[onboarding guide](framework-onboarding.md).

Run commands from `automation` with `.venv` activated. Start every suite with one
case on one model. Generation calls, judge calls and runtimes add up quickly on a
local machine, and no suite retries a failed answer.

## Developer API key

Create a key in AnythingLLM's API key settings and store it in a private file.
Settings read, in order:

1. a non-empty `ANYTHINGLLM_API_KEY` environment variable;
2. the file named by `ANYTHINGLLM_API_KEY_FILE`;
3. the local default `.runtime/anythingllm-api-key` at the repository root.

An explicitly configured missing key file is an error, not a silent fallback.
The key is excluded from the `Settings` repr. Never commit it or put it in
reports, screenshots or examples. API keys grant full access; they are not
read-only test credentials.

## Capturing the actual model context

Returned sources are citation evidence, not the context the generator received.
Semantic metrics need that context, so the capture overlay installs a preload
hook that observes the Ollama SDK `chat` call after AnythingLLM has compressed its
messages. Only requests marked by the test are captured, without authorization
headers. A capture write error fails the request instead of producing invented
evaluation inputs.

```bash
mkdir -p .runtime/ollama-capture
docker compose -f compose.yaml -f compose.capture.yaml up -d anythingllm
cd automation
python -m pytest tests/rag/test_rag.py -k paid_leave --rag-model qwen3.5:4b --capture-rag \
  --junitxml=reports/capture/results.xml
```

The sample is written to `reports/rag-samples/<capture-id>.json`. Restore the
ordinary service afterwards with `docker compose -f compose.yaml up -d anythingllm`.
Capture supports one non-streaming answer per temporary workspace; the UI's
streaming flow is not captured.

## Semantic metrics on a captured sample

All metrics use a local Qwen3.5 4B judge with structured JSON output, temperature
0, thinking disabled and a fixed seed. Each has a call budget and rejects
truncated or malformed judge output without retrying. Install
`requirements-evaluation.lock` first.

| Metric | What it measures | Judge calls | Command module |
| --- | --- | ---: | --- |
| Faithfulness | Share of answer claims supported by the captured contexts | ≤ 2 | `evaluation.faithfulness` |
| Factual correctness | Claim-level F1 against the golden reference | ≤ 4 | `evaluation.correctness` |
| Context precision/recall | Usefulness of retrieved contexts and coverage of the reference | ≤ 5 | `evaluation.relevance` |

```bash
python -m evaluation.faithfulness reports/rag-samples/<capture-id>.json \
  --judge-model qwen3.5:4b --output reports/evidence/faithfulness.json
python -m evaluation.correctness reports/rag-samples/<capture-id>.json \
  --case paid_leave --judge-model qwen3.5:4b --output reports/evidence/correctness.json
python -m evaluation.relevance reports/rag-samples/<capture-id>.json \
  --case paid_leave --output reports/evidence/relevance.json
```

### Hand-labelled judge controls

A judge score is trusted only after the judge agrees with known labels. The
controls in `data/faithfulness-controls.json` and
`data/correctness-controls.json` include supported paraphrases, wrong
numbers, invented amounts, incomplete answers and working/calendar-day
substitutions. Each claim must map to exactly one extracted statement with the
expected verdict; a matching average score is not enough.

```bash
python -m evaluation.calibration reports/rag-samples/<capture-id>.json \
  --controls data/faithfulness-controls.json \
  --control supported_answer --control wrong_numbers --control unsupported_gym_amount \
  --judge-model qwen3.5:4b --output reports/judge-controls.json
python -m evaluation.correctness reports/rag-samples/<capture-id>.json \
  --case paid_leave --control incomplete --output reports/correctness-controls.json
```

`evaluation.judge_validation` runs the wider source-bound control set
in `data/judge-validation.json` for all four metrics (use `--dry-run` to see
the plan and `--control … --max-judge-calls N` to narrow it).
`reporting.judge_review <benchmark.json>` lists, offline, the cases where
the judge and the reference disagree, for human review.

## Combined quality report and gates

`tests/quality/test_quality.py` turns saved evidence into an Allure report with
independent dimensions: required facts, sources, faithfulness, correctness and
context relevance. Every judge report is bound to the sample by SHA256. Without
`--quality-gates`, scores are recorded as `measured`; with it, the thresholds in
`data/quality-gates.json` apply. No model is called.

```bash
python -m pytest tests/quality/test_quality.py \
  --quality-sample reports/evidence/sample.json \
  --faithfulness-report reports/evidence/faithfulness.json \
  --correctness-report reports/evidence/correctness.json \
  --relevance-report reports/evidence/relevance.json \
  --quality-gates data/quality-gates.json \
  --alluredir reports/quality/allure-results --junitxml reports/quality/results.xml
npm --prefix tools/allure ci
npm --prefix tools/allure exec -- allure generate reports/quality/allure-results \
  --output reports/quality/allure-report
```

`tests/quality/test_live_quality.py` does the whole chain for one answer — setup,
capture, generation, faithfulness, report and verified cleanup — with at most
three model calls. It needs the capture overlay:

```bash
python -m pytest tests/quality/test_live_quality.py --run-live-quality --capture-rag \
  --rag-model qwen3.5:4b --judge-model qwen3.5:4b \
  --alluredir=reports/live/allure-results --clean-alluredir --allure-no-capture \
  --junitxml=reports/live/results.xml
```

## Golden dataset

`data/golden-policy.json` holds 16 versioned, source-bound cases: facts,
multi-part questions, boundaries and missing information. Each has a reference,
required and forbidden patterns and source anchors. Its version and checksum are
recorded with every run. The full suite is 16 generations per model.

```bash
python -m pytest tests/rag/test_golden_rag.py --run-golden --rag-model qwen3.5:4b -k carryover_limit \
  --junitxml=reports/golden/results.xml
python -m reporting.stability reports/golden/results.xml --output reports/golden/stability.json
```

Repeat with `--rag-repeat N` to measure variability; each repetition gets fresh
resources. The stability report compares outcomes of identical configurations.

## Regression suites built on the golden cases

All three reuse golden expectations unchanged, so a different outcome comes from
the changed input, not from relaxed criteria. Comparison tools treat missing or
uncontrolled runs as incomplete.

| Suite | Changes | Enable | Comparison tool |
| --- | --- | --- | --- |
| Prompt regression | Versioned system prompts from `prompt-variants.json` | `--run-prompt-regression` (`--rag-prompt ID` to select) | `reporting.prompt_regression` |
| Adversarial | Six attacks in the question or the document (`adversarial-policy.json`) | `--run-adversarial` | — |
| Counterfactual bias | Employee descriptors in three pairs (`bias-policy.json`) | `--run-bias` | `reporting.bias` |

```bash
python -m pytest tests/regression/test_prompt_regression.py --run-prompt-regression --rag-model qwen3.5:4b \
  -k carryover_limit --junitxml=reports/prompt-regression/runs.xml
python -m reporting.prompt_regression reports/prompt-regression/runs.xml \
  --case carryover_limit --model qwen3.5:4b --output reports/prompt-regression/comparison.json

python -m pytest tests/regression/test_adversarial.py --run-adversarial --rag-model qwen3.5:4b -k user_override \
  --junitxml=reports/adversarial/runs.xml

python -m pytest tests/regression/test_bias.py --run-bias --rag-model qwen3.5:4b -k gender_carryover \
  --junitxml reports/bias/runs.xml
python -m reporting.bias reports/bias/runs.xml --pair gender_carryover --model qwen3.5:4b \
  --output reports/bias/comparison.json
```

A document-injection attack counts only if capture proves the attack text reached
the model context; a citation alone is not proof.

## Dataset benchmark

`evaluation.benchmark_runner` evaluates selected golden cases with all
four metrics plus the faithfulness controls, serially and with a hard call budget.
The default four curated cases on one model allow at most 43 calls; two models
allow at most 80. Refusal cases are judged by their answer rules; their semantic
metrics are N/A, not zero or one. It needs the capture overlay.

```bash
python -m evaluation.benchmark_runner --output reports/benchmark --dry-run
python -m evaluation.benchmark_runner --output reports/benchmark
python -m evaluation.benchmark_runner --model qwen3.5:4b --model qwen2.5:7b \
  --max-model-calls 80 --output reports/benchmark-two-models
python -m pytest tests/quality/test_benchmark_report.py --benchmark-report reports/benchmark/benchmark.json \
  --alluredir reports/benchmark/allure-results --allure-no-capture --junitxml reports/benchmark/results.xml
```

## Fresh AI evidence in CI

The manual workflow `Fresh RAG quality` (`rag-live-quality.yml`) starts disposable
AnythingLLM and Ollama services on a GitHub-hosted runner, checks the API
contract, runs the benchmark and saves the evidence. A separate job revalidates
that evidence with zero model calls. Only if both succeed does the workflow
publish the required commit status `RAG benchmark quality`. A `smoke` profile
publishes a different status, so it cannot satisfy the merge gate.

| Profile | Models | Fresh answers | Maximum model calls |
| --- | --- | ---: | ---: |
| `smoke` | `primary` | 2 | 19 |
| `smoke` | `comparison` | 4 | 32 |
| `curated` | `primary` | 4 | 43 |
| `curated` | `comparison` | 8 | 80 |

```bash
gh workflow run rag-live-quality.yml --ref main -f pull-request=<number> -f profile=curated -f models=primary
```

The PR mode tests and marks the PR's merge commit. Any new commit needs a new
run. Hosted runners use CPU inference, so a curated run takes roughly two hours
and its timings are not comparable with local Apple Silicon results.

## Metric history and performance

`reporting.drift` records provenance-bound measurements and compares a
new record with an explicit baseline. A different case, policy, dataset, prompt,
judge or retrieval configuration makes the pair `incomparable`; a changed
generation model is allowed and flagged, since comparing models is a valid use.

```bash
python -m reporting.drift record --sample reports/evidence/sample.json \
  --faithfulness reports/evidence/faithfulness.json --correctness reports/evidence/correctness.json \
  --relevance reports/evidence/relevance.json
python -m reporting.drift compare reports/history/<baseline>.json reports/history/<current>.json \
  --maximum-drop 0.05 --output reports/drift/comparison.json
```

`tests/performance/test_performance.py` runs bounded health or RAG batches with per-attempt
timings and a declared p95 limit. It is never part of ordinary CI.

```bash
python -m pytest tests/performance/test_performance.py --run-performance --performance-mode health \
  --performance-requests 6 --performance-users 2 --performance-p95 5
python -m pytest tests/performance/test_performance.py --run-performance --performance-mode rag --rag-model qwen3.5:4b \
  --performance-requests 1 --performance-users 1 --performance-p95 180
python -m performance.comparison reports/performance/<baseline>.json \
  reports/performance/<current>.json --maximum-growth 0.2 --output reports/performance/comparison.json
```

## Conversation profile

`tests/rag/conversation` checks the Chat-mode workspace used for manual work:
greetings, small talk and mixed policy questions, five sequential generations.

```bash
python -m pytest tests/rag/conversation --run-conversation --rag-model qwen3.5:4b
```

## Qualification evidence (educational IQ/OQ/PQ)

`data/qualification-plan.json` maps requirements to pytest selectors and
declared matrices, and pins the checksums of the data they depend on. The
packager rebuilds the trace from JUnit, plan, data and sources, keeping failures
and missing cells. See the [test plan](test-plan.md) for the requirements.

```bash
python -m pytest tests/environment/test_installation.py tests/environment/test_health.py --junitxml reports/qualification/iq.xml
python -m pytest tests/api/test_authentication.py tests/api/test_documents.py tests/api/test_workspace.py \
  -k 'not configured_workspace' --junitxml reports/qualification/oq.xml
python -m qualification.package --phase IQ --phase OQ \
  --junit reports/qualification/iq.xml --junit reports/qualification/oq.xml \
  --output reports/qualification/iq-oq-package
python -c "from pathlib import Path; from qualification.package import verify_package; \
verify_package(Path('reports/qualification/iq-oq-package'))"
```

PQ uses the golden, prompt, adversarial, bias, UI, quality and performance suites
above. Their complete matrices alone need about 120 generations, so the focused
examples in this guide do not amount to a full PQ. This is an educational format,
not GxP approval.
