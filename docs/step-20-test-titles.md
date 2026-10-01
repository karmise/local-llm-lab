# Step 20: Explicit Allure test titles

Every test function has an explicit English display title. Tests use
`@title("...")` from the existing reporting bridge, which delegates to the native
`allure.title` decorator when Allure is installed. Without the optional reporting
package, this decorator returns the original test unchanged so offline CI still
works. Step contexts and attachment logic remain outside test bodies.

Titles describe expected behavior instead of repeating Python function names.
Parameterized scenarios include `[{param_id}]`, which Allure resolves using the
pytest case ID. This distinguishes model names, repeated RAG runs and individual
unit-test variants without inserting raw input payloads into titles.

Examples:

- `Developer API rejects an invalid API key`
- `UI conversation keeps its question and answer after reload`
- `RAG answer contains paid-leave facts and supporting sources [qwen3.5:4b]`
- `Vector-search check rejects incomplete or unrelated results [wrong-document]`

Saved and live quality tests have separate titles. The quality presentation helper
no longer replaces a test's declared title dynamically; it continues to provide
feature/story metadata, diagnostic steps and evidence attachments.

## Validation

All 61 test functions have a declared title. Collection of the default suite
produces 95 distinct resolved display names, including all parameter variants.
Verification of actual Allure results passed 86 unit/API/saved-quality cases;
four UI cases and the live quality case stayed explicitly skipped, and the four
ordinary RAG cases were deselected. No model generation was needed for this change.
The 78 unit checks also passed with pytest plugin auto-loading disabled.

From `automation`, generate a fresh report after a suitable test run:

```bash
python -m pytest tests/unit --alluredir reports/test-titles/allure-results
npm --prefix tools/allure exec -- allure generate reports/test-titles/allure-results \
  --output reports/test-titles/allure-report
```
