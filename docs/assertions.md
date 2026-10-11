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

| Module | Checks |
| --- | --- |
| `api` | `assert_online`, `assert_api_key_accepted`, `assert_api_key_rejected`, `assert_created_workspace`, `assert_workspace_matches`, `assert_workspace_absent`, `assert_operation_success`, `assert_uploaded_document`, `assert_embeddings_updated`, `assert_workspace_document_attached`, `assert_search_contains`, `assert_document_folder_absent`, `assert_model_available`, `assert_models_available`, `assert_supported_runtime` |
| `answers` | `assert_completed_answer`, `assert_rag_answer`, `assert_required_facts`, `assert_document_sources`, `assert_missing_policy_information`, `assert_missing_policy_answer`, `assert_golden_answer`, `assert_golden_text`, `assert_conversation_answer`, `assert_adversarial_answer`, `assert_adversarial_context`, `assert_attack_exposure`, `assert_bias_answer` |
| `quality` | `assert_quality_score`, `assert_calibration_result`, `assert_quality_dimension`, `assert_quality_report`, `assert_benchmark_report`, `assert_benchmark_case`, `assert_performance_batch` |
| `ui` | `assert_ui_question_visible`, `assert_ui_completed_answer`, `assert_ui_policy_answer`, `assert_ui_document_source`, `assert_ui_source_content`, `assert_ui_history_preserved` |

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

Answer checks look for required facts, forbidden content and sources, and
exclude the model's thinking from fact matching. `assert_required_facts` is shared
by live RAG checks and saved-answer reports, so both apply the same rule.

Quality checks validate evidence, not thresholds: `assert_quality_score` only
requires a finite score between 0 and 1, and quality gates apply minimums
separately. `assert_calibration_result` requires every labelled control claim to
map to exactly one extracted statement with the expected verdict; matching average
scores are not enough. `assert_quality_dimension` and `assert_quality_report`
validate independent outcomes without averaging them.

Unit tests for these checks live in `tests/unit/test_assertions.py` and use plain
pytest, not this package, so a broken check cannot verify itself. Add new
repeated checks to the relevant module with type annotations and descriptive
messages; avoid dumping complete HTTP responses or headers in diagnostics.

[pytest assertion rewriting documentation](https://docs.pytest.org/en/stable/how-to/assert.html#assertion-introspection-details)
