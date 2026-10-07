# Step 22: Source-bound golden acceptance dataset

## Purpose and scope

The framework now has a versioned golden dataset for the fictional company
policy. Expected answers are authored from that document rather than copied from
model generations. This is a curated acceptance benchmark for this small policy,
not a statistically representative clinical dataset or a measurement of general
model accuracy. No patient data is included.

The catalog contains 16 cases across four categories:

| Category | Cases | Coverage |
| --- | --- | --- |
| `fact_lookup` | 5 | Leave approval, carryover limit/deadline, travel allowance and expense deadline |
| `multi_fact` | 5 | Paid leave and notice, hotel amount/receipt, travel approvers, remote-work limit/approval and availability/time zone |
| `boundary` | 3 | Incorrect calendar-day premise, carryover above the maximum and hotel receipt condition |
| `missing_information` | 3 | Gym reimbursement, bonus and parental leave |

Each case includes a stable ID, category, question, reference answer, rationale,
required and forbidden regex criteria, and exact supporting policy fragments.
Required patterns accept selected paraphrases and numeric formats; references
are not compared as complete strings. Sources must cite the uploaded test
file and contain the configured supporting fragments, including for missing-policy
questions. This source requirement is a deliberate acceptance contract.

The original two RAG scenarios and API/UI profiles remain available as a small
baseline. An offline check keeps the golden paid-leave question, reference and
source fragments aligned with the existing paid-leave profile.

## Files and responsibilities

- [golden-policy.json](../automation/test_data/golden-policy.json): expectations.
- [datasets/golden.py](../automation/src/llm_testkit/datasets/golden.py): frozen
  typed cases and validation, with no HTTP or model calls.
- [test_golden_rag.py](../automation/tests/test_golden_rag.py): one parameterized
  scenario consuming existing resource/model fixtures and Assertions.
- `pytest_support/options.py`: catalog collection, model/repetition matrix and
  explicit enabling.
- `test_support/fixtures/environment.py`: dataset provenance recorded as JUnit properties.
- `assertions.assert_golden_answer`: completed final answer, required/forbidden
  content and sources; a static Allure step and explicit answer attachment.
- `reporting/stability.py`: preserves golden case identity and dataset fingerprint.

The loader verifies schema, unique IDs, categories, nonempty fields, regex
validity, reference/rule consistency and source fragments against the policy.
The catalog stores the policy SHA256: changing the document fails validation until
expectations are reviewed. A checksum ties versions together; it does not prove
semantic correctness or expert review. Cases and their nested collections are
immutable after loading.

Every run records dataset version/checksum, case ID and category alongside the
existing model digest, document checksum, workspace configuration and repetition.
Optional captured samples retain these golden metadata fields too. Stability
summaries group by case and model, and treat changed dataset checksums as different
configurations rather than attributing all variation to a model.

## Controlled execution

Commands run from `automation`, using the configured virtual environment.

Start offline:

```bash
python -m pytest tests/unit -q
python -m pytest tests/test_golden_rag.py --collect-only --rag-model qwen3.5:4b
```

Without `--run-golden`, golden scenarios skip before resource/model fixtures run.
The dataset is still validated during collection. Existing default API/RAG behavior
remains; a bare pytest run is not an offline run.

Start with one case and one model against a running application:

```bash
python -m pytest tests/test_golden_rag.py --run-golden \
  --rag-model qwen3.5:4b -k carryover_limit \
  --junitxml=reports/golden-single/results.xml
```

Run one category using case IDs, for example missing-information cases:

```bash
python -m pytest tests/test_golden_rag.py --run-golden \
  --rag-model qwen3.5:4b -k missing
```

Run the full dataset on one model only when the additional generation cost is
intentional:

```bash
python -m pytest tests/test_golden_rag.py --run-golden \
  --rag-model qwen3.5:4b --junitxml=reports/golden-full/results.xml
```

That command generates 16 answers. Omitting `--rag-model` uses both default models
and produces 32 cases. `--rag-repeat N` multiplies those counts; each case owns a
fresh workspace and document folder. There are no automatic generation retries
and no judge calls in these golden acceptance tests. Embedding/indexing also
uses local resources. Use a new report directory for each run.

With reporting dependencies installed, add `--alluredir` to obtain readable test
titles and answer/step evidence. With the capture overlay installed, optional
`--capture-rag` preserves actual non-streaming model context. Neither option is
required for ordinary golden acceptance execution.

The existing stability utility consumes the JUnit output:

```bash
python -m llm_testkit.reporting.stability reports/golden-full/results.xml \
  --output reports/golden-full/stability.json
```

## Maintaining expectations

When adding a case, derive the reference from the policy, state the behavior being
protected in `rationale`, and choose source fragments that justify that behavior.
Include required facts and, where useful, forbidden content. Run the offline
reference/validation tests before generating any new model answers.

When changing the policy, review every affected reference, criterion and source
fragment, update the catalog version and policy checksum, then rerun validation.
Do not update the checksum merely to bypass a failing check or derive expected
facts from whatever a model happened to output.

## Interpretation and limitations

Passing means the configured acceptance rules and source checks passed for that
case. Regexes can miss negation, contradiction or unanticipated phrasing. A boundary
case is not a comprehensive reasoning evaluation. Missing-information forbidden
patterns cover selected amounts/durations, not every way of inventing a benefit.
The loader checks consistency but cannot establish semantic correctness of labels.

These cases are visible to developers and are not an unseen holdout set. There
is no dataset-level calibrated AI quality threshold, semantic correctness metric,
retrieval precision/recall metric or full adversarial suite in this stage. The
existing faithfulness evaluation remains separate and has no calibrated gate.
The next evaluation stage can use this dataset as a foundation without changing
the meaning of these deterministic acceptance checks.

## Validation

On 2026-10-06, all 179 unit tests passed, including catalog consistency, rejected
invalid expectations, forbidden benefits, incomplete/source failures and golden
collection/reporting behavior. Ruff passed. Without enabling golden execution,
all 16 cases on one selected model skipped before external setup.

One live `carryover_limit` case on `qwen3.5:4b` passed in 32.20 seconds, including
source checks and verified resource cleanup. The other 15 references and rules
were checked offline; the full live catalog was not executed in this stage.
No judge calls were made. Local results are ignored under
`automation/reports/golden-smoke/`.
