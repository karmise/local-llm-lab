# Step 9: Repeated-Run Stability

RAG scenarios support independent repetitions through `--rag-repeat`.
Each scenario/model/repetition gets a new workspace and document folder.
Failures are retained; later repetitions do not replace them with a successful
retry. The default remains one run per model and scenario.

## Execute a small experiment

From `automation`, with AnythingLLM and Ollama running:

```bash
source .venv/bin/activate
python -m pytest -m rag --rag-repeat 3 -v --junitxml=reports/stability.xml
python -m llm_testkit.reporting.stability reports/stability.xml --output reports/stability.json
```

Two scenarios × two models × three repetitions produces twelve RAG cases.
Select a model with `--rag-model qwen2.5:7b` or `--rag-model qwen3.5:4b`.
Invalid, zero and negative repetition counts are rejected before execution.
Test IDs include the model and repetition when repeating, for example:

```text
test_paid_leave_answer_is_grounded_in_policy[qwen3.5:4b-run-2]
```

JUnit properties retain the model digest, policy hash, workspace configuration,
thinking mode and repetition number. Resource names are unique while policy
content, questions and acceptance rules remain the same.

## Offline summary

The summary groups results by scenario and model. Each row contains:

- `runs`, `passed`, `failed`, `errored`, `skipped`
- `metadata_complete`
- `configuration_consistent`
- `mixed_pass_fail_observed`

Configuration consistency compares the model digest, policy hash, normalized
workspace configuration and recorded thinking mode. Missing metadata or a
changed fingerprint prevents a claim of consistent recorded configuration.

Setup and teardown errors are counted separately. If pytest emits separate
call-failure and teardown-error XML entries for the same test ID, they count as
one run with both signals. Cases failing before metadata recording are recovered
from the known RAG test IDs and marked as incomplete rather than ignored.

`mixed_pass_fail_observed` means that both passed and failed executions were
observed in the group. It is a review signal, not an automatic explanation or
a flaky-test verdict. Call-phase transport failures can also appear as failures;
inspect the original JUnit diagnostics before attributing them to answer quality.
The summary covers recorded inputs, not complete model context, process load or
an explicitly fixed model seed.

This utility reads one report and has no network access. It does not hide
failure messages, rerun tests or change pytest's exit status. The original XML
remains the authoritative record for individual failures and cleanup errors.

## Unit coverage

Four new unit tests verify mixed outcomes, changed configuration, setup errors
without metadata and duplicate call/teardown entries. Checks still use the
shared assertions module. The framework unit suite now contains 24 cases.

## Interpretation

Three passes mean no failure was observed in those three executions. They do not
prove that a scenario is never flaky or establish general model accuracy.
Consistent failures on one model differ from pass/fail variation on the same
recorded configuration. Investigation should consider assertion wording,
retrieval, response parsing, model behavior and infrastructure.

If failures appear, preserve their reports and inspect the final answer and
supporting sources before changing acceptance rules. Do not relax rules solely
to make a model pass.

Reports and JSON summaries remain under `automation/reports/`, excluded from Git.

## Recorded experiment

On 2026-10-01, all 12 repeated RAG cases passed in 232.73 seconds, with no
setup or teardown errors. The summary confirms complete metadata and one
recorded configuration fingerprint within each scenario/model group.

| Scenario | Qwen3.5 4B | Qwen2.5 7B |
| --- | --- | --- |
| Paid leave facts and sources | 3/3 passed | 3/3 passed |
| Missing gym policy | 3/3 passed | 3/3 passed |

No mixed pass/fail outcomes were observed. This is a small initial experiment,
not evidence that flaky behavior is impossible. Run durations include resource
setup, model loading, generation and teardown and are not inference benchmarks.

The 24 unit checks and 7 API/health checks also passed in separate verification
runs. Invalid repetition inputs were verified to produce configuration errors.
Local artifacts: `automation/reports/step-09-stability.xml` and
`automation/reports/step-09-stability.json`.
