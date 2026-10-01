# Step 7: Missing Policy Information

The second RAG scenario asks about gym membership reimbursement, which the
fictional policy explicitly excludes from its scope. It checks that the final
answer acknowledges the missing policy instead of supplying an amount.

## Scenario

`test_missing_gym_policy_does_not_invent_reimbursement` asks:

> What is the company's gym membership reimbursement policy, and how much does it reimburse per month?

`assert_missing_policy_information` checks:

1. A successful, completed chat response without an application error.
2. An explicit missing-information statement in the final answer.
3. A reference to gym or fitness reimbursement in that answer.
4. No recognized currency amount, recurring amount or reimbursement percentage.
5. A source from the uploaded test document containing its gym-policy scope
   statement.

Recognized amount forms include `KGS 500`, `500 som`, `five hundred som`,
`500 per month`, currency symbols and percentages. Ordinary section numbers
are allowed: citing section 4 is not a reimbursement claim.

This is a rule-based check with documented phrase and amount patterns. It is not
a general semantic or hallucination detector. Unsupported valid wording may
require extending the acceptance rules; a failure should be inspected before
being labeled a model regression or a flaky result.

## Shared answer processing

`assert_completed_answer` now performs the shared response contract checks and
returns the checked payload and normalized final answer. Both RAG validators
use it, so thinking blocks are consistently excluded. A missing-information
statement present only inside reasoning cannot pass the scenario.

`assert_document_sources` centralizes validation of the uploaded source and
its supporting passages. The missing-information test requires the explicit
scope passage, because the fixture contains it. A query-mode short-circuit
with no retrieved sources cannot pass as evidence of a grounded model answer.

The general clients and resource fixtures are unchanged. Each RAG test owns
its own indexed workspace and document folder; the existing teardown removes
them after execution.

## Unit coverage

Six additional cases exercise:

- Missing-information wording followed by an invented currency amount.
- An invented recurring amount without a currency.
- A written-number currency amount.
- An abstention statement appearing only in reasoning.
- An answer addressing the wrong topic.
- A valid scope statement with a harmless document section number.

These unit cases use synthetic responses and do not call a model.

## Run

From `automation`, with the local application and Ollama running:

```bash
source .venv/bin/activate
python -m pytest tests/test_rag.py::test_missing_gym_policy_does_not_invent_reimbursement -v
python -m pytest -m rag -v
python -m pytest -m 'not rag' -v
```

## Verification

Recorded on 2026-10-01 with Qwen3.5 4B and the existing controlled configuration.
The first standalone gym scenario passed in 24.12 seconds.
The non-RAG suite passed all 25 selected checks in 0.56 seconds (7 API/health
and 18 unit checks). These are pytest run durations, not model benchmarks.
Reports remain local under `automation/reports/` and excluded from Git.

After the final shared-validator changes, both RAG scenarios passed in a
separate run: **2 passed in 65.38 seconds**, including successful teardown.
Together with the 25 non-RAG checks, all **27 current checks passed**.
The generation report is `automation/reports/step-07-rag.xml`.

## Next step

Run both RAG scenarios with Qwen3.5 4B and Qwen2.5 7B using the same document,
prompt settings and acceptance rules. Record the model configuration with each
result, and distinguish consistent model differences from changes across
repeated runs of the same model.
