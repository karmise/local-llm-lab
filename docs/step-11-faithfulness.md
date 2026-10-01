# Step 11: Local RAGAS Faithfulness Evaluation

## Metric and inputs

RAGAS 0.4.3's modern `metrics.collections.Faithfulness` extracts atomic statements
from the saved response and checks each against the actual document contexts.
Its score is supported statements divided by evaluated statements. Inputs are
the question, final response and contexts. The reference answer is retained for
future metrics but is not an input to faithfulness.

This is model-based judgement, not a deterministic fact check or an overall
measurement of correctness, completeness or retrieval quality. Existing
scenario assertions still check required facts and citation sources.

## Install and run

From `automation`:

```bash
.venv/bin/python -m pip install -e '.[evaluation]' -c requirements-evaluation.lock
.venv/bin/python -m llm_testkit.evaluation.faithfulness \
  reports/rag-samples/<capture-id>.json \
  --judge-model qwen3.5:4b \
  --output reports/faithfulness-<capture-id>.json
```

Use a sample from [step 10](step-10-context-capture.md) and an installed judge
model. Evaluation uses native Ollama directly; AnythingLLM does not generate
another answer and the capture overlay is not needed. Ordinary pytest runs do
not invoke a model judge. Optional adapter unit checks use fake responses only.

The evaluation extra pins compatible LangChain versions because RAGAS 0.4.3
imports legacy VertexAI modules removed in Community 0.4. The evaluation lock
records the dependency profile on Python 3.12.3/macOS ARM64. The base lock uses
packaging 25.0 for compatibility with the evaluation installation.

## Layers and execution limits

`HttpClient` handles transport, and `OllamaClient.structured_chat` exposes the
native operation. `evaluation.OllamaJudge` implements RAGAS's structured
generation interface using Ollama JSON schema and Pydantic validation. The
faithfulness service runs the unmodified RAGAS metric and checks its outputs.
Score assertions remain in `assertions.py`.

Judge settings: temperature 0, seed 42, context limit 8192, output limit 2048,
`think=false`, non-streaming output, at most two calls, no automatic retries.
HTTP timeout uses `ANYTHINGLLM_LLM_TIMEOUT` (default 300 seconds). These settings
reduce variability but do not guarantee reproducibility. The limits target
these small scenarios, not arbitrary long documents.

Incomplete/truncated responses, invalid JSON/schema, no claims, NaN scores,
missing/changed/duplicated claim verdicts and nonbinary verdicts fail evaluation.
Failures produce `status=error` and exit code 1, not a zero score or a passing
result. Successful measurements use `status=completed` and exit code 0. No
quality threshold is enabled yet: completion is not a quality gate.

## Reports and limitations

Reports contain the sample checksum, RAGAS version, generator/judge names,
judge digest/settings, extracted claims, individual verdicts/reasons, judge
prompts and raw responses. Files are local, ignored by Git, created with
owner-only permissions and never overwritten. Saved contexts are checked
against their observation before model calls.

RAGAS analytics are disabled before import. The adapter uses only `OLLAMA_BASE_URL`
(default localhost:11434), without cloud credentials. OpenAI/LangChain packages
are transitive dependencies; their cloud clients are not used. Judge prompts
contain policy data and should be handled like captured samples.

The report records when generator and judge use the same model. This may cause
correlated errors and self-evaluation bias. Small models can miss claims or
misjudge support even with valid JSON. Coverage checks ensure all *extracted*
claims receive verdicts; they cannot prove extraction found every original claim.

## Verified experiment

On 2026-10-01, the saved paid-leave sample was evaluated using Qwen3.5 4B.
RAGAS extracted two claims: 23 working days, and requests at least 12 calendar
days before leave. Both verdicts were 1, giving faithfulness 1.0. The two Ollama
calls took 13.30 seconds in total according to server durations. This is a
result for one sample and judge configuration, not an accuracy estimate.

All 49 framework unit checks passed, including the real RAGAS pipeline with
fake responses, a supported/unsupported claim ratio of 0.5, malformed results,
claim coverage, truncation and score validation. Dependencies had no conflicts.

Continue with [step 12](step-12-judge-controls.md) to check hand-labelled
correct answers, swapped numbers and unsupported claims before selecting a
production quality threshold.

## Primary references

- [RAGAS faithfulness definition and API](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)
- [Pinned metric implementation](https://github.com/vibrantlabsai/ragas/blob/v0.4.3/src/ragas/metrics/collections/faithfulness/metric.py)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama chat options](https://docs.ollama.com/api/chat)
