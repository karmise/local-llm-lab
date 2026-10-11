# Reusable Assertions

All explicit checks live in the `automation/src/llm_testkit/assertions/` package, one module per
area (`fields`, `api`, `answers`, `quality`, `ui`); `from llm_testkit import assertions` exposes all of them.
Tests describe scenarios and call named checks. Fixtures also use this module
to validate resource setup and cleanup. API clients still return responses
without checking expected outcomes.

## General checks

| Function | Purpose |
| --- | --- |
| `assert_status_code(response, expected, context=...)` | HTTP status with operation context |
| `assert_json_object(response)` | Valid JSON object; returns the checked body |
| `assert_field_type(payload, field, expected)` | Required field with exact runtime type; returns its value |
| `assert_field_equals(payload, field, expected)` | Field type and equality |
| `assert_field_length(payload, field, expected)` | Length of a collection or string |
| `assert_field_contains(payload, field, expected)` | Required substring in a string field |
| `assert_field_starts_with(payload, field, expected)` | String prefix, including temporary resource names |

Strict type checks distinguish JSON `true` from `1`, and integer IDs from
booleans. Equality checks use exact types and values; future numeric tolerance
or nested schema rules should be explicit checks rather than implicit coercion.
Malformed JSON produces an assertion failure with a clear message.

## Application checks

- `assert_online(response)`
- `assert_api_key_accepted(response)`
- `assert_api_key_rejected(response)`
- `assert_created_workspace(response, expected_name)`
- `assert_workspace_matches(response, slug=..., workspace_id=..., name=..., configuration=...)`
- `assert_workspace_absent(response, slug)`
- `assert_operation_success(response, context=...)`
- `assert_uploaded_document(response, folder=..., filename=...)`
- `assert_embeddings_updated(response, slug=...)`
- `assert_workspace_document_attached(response, slug=..., location=...)`
- `assert_search_contains(response, document_title=..., fragments=...)`
- `assert_document_folder_absent(response, folder=...)`
- `assert_rag_answer(response, fact_patterns=..., document_title=..., source_fragments=...)`
- `assert_completed_answer(response)`
- `assert_document_sources(payload, document_title=..., fragments=...)`
- `assert_missing_policy_information(response, document_title=..., source_fragments=...)`
- `assert_model_available(models, name)`

These compose the general checks and express the AnythingLLM contract.
`assert_created_workspace` returns the checked workspace for fixture ownership;
the other scenario checks return `None`. A failed check raises `AssertionError`.

Example test validation:

```python
response = authenticated_anythingllm_api.get_workspace(temporary_workspace["slug"])
assertions.assert_workspace_matches(
    response,
    slug=temporary_workspace["slug"],
    workspace_id=temporary_workspace["id"],
    configuration=workspace_configuration,
)
```

## pytest diagnostics and verification

`tests/conftest.py` registers `llm_testkit.assertions` for pytest assertion
rewriting before importing it. This preserves detailed assertion diagnostics
inside the shared module. Registration belongs to the pytest layer; importing
the framework package does not itself import pytest.

Using direct assertions in pytest tests is valid. This project chooses reusable
checks to centralize repeated contracts and keep scenario code concise.

Verified after refactoring: all six application checks passed. Four unit tests
exercise invalid inputs: integer instead of boolean, missing field, malformed
JSON, and a non-object workspace entry. Unit tests require no application or key.

```bash
python -m pytest -m unit -v
python -m pytest -m 'api or smoke' -v
```

Document retrieval adds three unit cases for empty results, the wrong source
document and a missing required fact. The current suite contains seven unit
checks; see step 5 for application verification results.

RAG validation adds five negative unit cases and excludes thinking from answer
fact checks; see [step 6](step-06-rag-answer.md) for details. The current unit
suite contained twelve checks at that step. [Step 7](step-07-missing-information.md)
added six missing-information cases, bringing the unit suite to eighteen checks at that step.

[Step 8](step-08-model-comparison.md) adds two model-selection cases (missing
model and missing digest); the current unit suite contains twenty checks.

New repeated checks should be added here with type annotations and descriptive
messages. Avoid dumping complete HTTP responses or headers in diagnostics.

[Step 11](step-11-faithfulness.md) adds `assert_quality_score(value)`; thresholds are applied by quality gates.
It rejects nonnumeric, nonfinite and out-of-range values. An explicit minimum
can be supplied after judge calibration; no default quality gate is enabled.

[Step 12](step-12-judge-controls.md) adds `assert_calibration_result` to check
hand-labelled control scores and individual claim verdicts. Claims must map
one-to-one to extracted statements; matching average scores alone are insufficient.

[Step 14](step-14-allure-quality-report.md) shares `assert_required_facts` between
live RAG checks and saved-answer reporting. `assert_quality_dimension` and
`assert_quality_report` validate independent outcomes without averaging scores
or inventing a default faithfulness threshold.

[pytest assertion rewriting documentation](https://docs.pytest.org/en/stable/how-to/assert.html#assertion-introspection-details)
