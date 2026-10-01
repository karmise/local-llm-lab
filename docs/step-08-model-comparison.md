# Step 8: Generation Model Comparison

Both RAG scenarios now run against Qwen3.5 4B and Qwen2.5 7B by default.
Test logic and acceptance criteria are shared; each case creates its own
workspace and document resources. The main Company Policy Lab is not switched.

## Parameterization

The pytest collection hook expands only tests marked `rag` using the
`generation_model` fixture. Other tests retain the default model from
`config/workspace.json` and are not duplicated.

Example test IDs:

```text
test_paid_leave_answer_is_grounded_in_policy[qwen3.5:4b]
test_paid_leave_answer_is_grounded_in_policy[qwen2.5:7b]
test_missing_gym_policy_does_not_invent_reimbursement[qwen3.5:4b]
test_missing_gym_policy_does_not_invent_reimbursement[qwen2.5:7b]
```

The workspace configuration fixture copies the template and changes only
`chatModel`. The same policy bytes, system prompt, temperature, history setting,
retrieval settings, question text and assertions are used across models.
Temporary resource names differ for isolation. The full context sent to the
model is not captured by this step.

## Preflight and result metadata

A small Ollama client uses the shared HTTP transport to read `/api/tags`.
`OLLAMA_BASE_URL` defaults to `http://127.0.0.1:11434`; set it to the actual
Ollama server used by AnythingLLM when running another environment.
The selected model must be installed and have a digest. Missing models fail
setup instead of silently skipping tests or triggering downloads.

Before generation, `rag_environment` verifies the configured workspace through
the AnythingLLM API. It records these per-test JUnit properties:

- `generation_model`
- `model_digest`, from the live model catalog captured for the run
- `policy_sha256`
- `workspace_configuration`
- `thinking_mode`

The report uses pytest's `xunit1` format to include custom testcase properties.
Secrets and HTTP authorization headers are not recorded. Reports remain local
under `automation/reports/` and are excluded from Git.

Thinking is still the Ollama/model default. The two models can use different
reasoning behavior; this is a comparison of the configured systems, not a claim
of identical compute budgets. No automatic retries are enabled.

## Run

From `automation`, with both generation models installed:

```bash
source .venv/bin/activate
python -m pytest -m rag -v --junitxml=reports/model-comparison.xml
```

Select one model:

```bash
python -m pytest -m rag --rag-model qwen2.5:7b -v
```

Select multiple models explicitly:

```bash
python -m pytest -m rag --rag-model qwen3.5:4b --rag-model qwen2.5:7b -v
```

Repeated identical model options are deduplicated; they do not create stability
repetitions. In PyCharm, put the same options in the pytest run configuration's
additional arguments. Running a single RAG test without a model option still
collects both default model variants.

## Verification

Recorded on 2026-10-01:

| Scenario | Qwen3.5 4B | Qwen2.5 7B |
| --- | --- | --- |
| Paid leave facts and sources | PASS | PASS |
| Missing gym policy without an invented amount | PASS | PASS |

The four model cases passed in 76.67 seconds, including resource setup and
teardown. Separately, all 7 API/health checks and 20 unit checks passed:
31 successful checks across the three verification runs. One-model collection
was also verified to contain exactly the two selected RAG scenarios.

The comparison XML was inspected: all four entries contain matching policy
hashes and workspace settings except `chatModel`, with the expected model
digests. Local report: `automation/reports/step-08-model-matrix.xml`.
These run durations are not inference benchmarks or evidence of model ranking.

## Next experiment

Repeat each model configuration without changing the input or acceptance rules.
Retain each run's report under a separate filename. A consistent difference
between models is distinct from an outcome changing across runs of one model.
The first successful matrix does not establish absence of flaky behavior.

## References

- [Ollama model catalog](https://docs.ollama.com/api/tags)
- [pytest JUnit properties](https://docs.pytest.org/en/stable/how-to/output.html#record-property)
