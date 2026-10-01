# Step 19: Reporting belongs to framework operations

Tests now describe actions and expected behavior without importing Allure or
opening Allure step contexts. UI actions report themselves from the Page Object,
UI checks report themselves from Assertions, and selected developer API operations
report themselves from the API client. UI feature/story metadata is assigned by
the UI fixture layer. Screenshots and displayed-answer attachments are preserved.

## Selective API reporting

The reported API operations are workspace creation/deletion, document upload,
workspace indexing/search, answer generation and test document-folder deletion.
Health, authentication, workspace lookup, folder lookup and folder creation remain
plain requests to keep the report readable. The HTTP transport is not instrumented.

Step titles are fixed strings. The reporting wrapper does not log function
arguments, request headers, API keys, response bodies or client objects. Reporting
does not add assertions or retries to clients, change their return values or
suppress exceptions. Fixture cleanup keeps the same lifecycle.

## Optional reporting bridge

`reporting/steps.py` provides a typed synchronous `@step` decorator and explicit
attachment helpers. It uses a context-managed Allure step rather than Allure's
argument-recording method decorator. If Allure is not installed, operations still
run normally. Other dependency errors and operation failures propagate unchanged.
`functools.wraps` preserves function metadata and the original callable.

The bridge has no pytest or mandatory browser dependency. UI failure screenshots
and traces are attached through the same bridge after Playwright retains them.
The captured live-quality operations moved into `reporting/live_quality.py`, and
the saved-quality reporter owns the original judge-evidence attachment.

## Verification

From `automation`:

```bash
python -m ruff check src tests --config pyproject.toml
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/unit
python -m pytest -m 'api or smoke' --alluredir reports/step-19/api-results
python -m pytest tests/ui --run-ui -k source_details \
  --tracing retain-on-failure --screenshot only-on-failure \
  --output reports/step-19/browser-artifacts \
  --alluredir reports/step-19/ui-results
```

The unit checks cover exception/result preservation, optional reporting and the
absence of automatic argument/parameter logging. The focused UI scenario checks
browser actions, source content, screenshots and API setup/cleanup with one model
generation. The saved quality report can be verified using existing sample and
judge evidence without additional model calls.

Validation completed: 78 unit tests, seven API/smoke checks, one real UI source
scenario and one saved quality report passed. A temporary intentional UI failure
confirmed a failed assertion step plus screenshot/trace attachments; the probe
was removed after verification. All four UI scenarios still skip by default.
