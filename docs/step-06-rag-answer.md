# Step 6: Grounded Policy Answer

The first generation test asks the model about annual paid leave and advance
notice for a leave request. It uses the English policy in a fresh, indexed
workspace. No changes are made to the main Company Policy Lab.

## Scenario and acceptance criteria

`test_paid_leave_answer_is_grounded_in_policy` sends a query-mode chat request:

> How many working days of paid leave does each employee receive per year, and how many calendar days before leave starts must a request be submitted?

`assert_rag_answer` checks:

1. HTTP 200 and a completed `textResponse` without an application error.
2. A non-empty final answer containing 23 working/business days and 12 calendar
   days before leave starts. Numeric and written-number variants are accepted.
3. A source from the current test's uniquely named document.
4. Source passages containing the reference facts `23 working days` and
   `12 calendar days`.

Thinking blocks enclosed by `<think>...</think>` are excluded. Facts present
only in reasoning cannot pass. Incomplete thinking tags fail validation.
Markdown emphasis and whitespace are normalized before checking answer facts.
Sources are checked separately against document text; reasoning is not a source.

The answer checks use explicit regular expressions rather than matching the
whole response. They are narrow acceptance rules, not a general semantic judge:
unlisted valid paraphrases may fail and arbitrary contradictions are not fully
analyzed. A successful run does not establish overall accuracy or hallucination
rates. Returned source passages also do not establish the entire model context
needed for RAGAS.

## Framework changes

- `AnythingLLMClient.chat(...)` calls POST `/api/v1/workspace/{slug}/chat` with
  `message` and `mode`.
- `Settings.llm_timeout` reads `ANYTHINGLLM_LLM_TIMEOUT`, defaulting to 300 seconds.
- `assert_rag_answer(...)` centralizes answer and source checks.
- The `rag` marker separates generated-answer tests from API-only scenarios.
- Existing fixtures own document and workspace cleanup, including chat history
  associated with the temporary workspace.

Configuration remains Qwen3.5 4B, temperature 0.1, query mode, history 0,
topN 4 and similarity threshold 0.25. The installed AnythingLLM connector uses
the model's default thinking behavior. Generation is not retried automatically.

The LLM timeout permits slower generation independently of health and document
timeouts. It remains a Requests connection/read timeout, not a total test deadline.

## Run

From `automation`, with AnythingLLM and Ollama running:

```bash
source .venv/bin/activate
python -m pytest -m rag -v
python -m pytest -m 'not rag' -v
python -m pytest -v
```

An alternative generation timeout can be supplied per run:

```bash
ANYTHINGLLM_LLM_TIMEOUT=180 python -m pytest -m rag -v
```

## Unit checks

Five new negative cases ensure the RAG checker rejects:

- A fact appearing only inside a thinking block.
- An incorrect paid-leave amount.
- A source belonging to another document.
- A cited document passage that does not support the required fact.
- An incomplete thinking block.

These cases use synthetic responses and require no model or application.

## Verification

The first standalone local generation scenario passed in 40.25 seconds on
2026-10-01. This is a single pytest run duration, not an inference benchmark or
a flakiness estimate. Full-suite results are recorded in the local
`automation/reports/step-06.xml` report, excluded from Git.

Full-suite verification after the final validation changes: **20 passed in
34.38 seconds** (7 API/health checks, 1 generated-answer scenario and 12 unit
checks). Document and workspace teardown completed without errors.

## Next step

Add a question about gym reimbursement, which the policy does not describe.
Check that the model reports insufficient information instead of inventing an
amount. Then execute the same criteria with both Qwen3.5 4B and Qwen2.5 7B and
record their model configurations for comparison.

[Installed-version chat API](https://github.com/Mintplex-Labs/anything-llm/blob/v1.16.2/server/endpoints/api/workspace/index.js)
