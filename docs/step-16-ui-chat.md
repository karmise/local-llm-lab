# Step 16: Workspace chat through the UI

The first browser scenario prepares an isolated workspace and indexes an English
policy through the developer API. It sends one paid-leave question through the
real AnythingLLM UI and checks the displayed question, the completed final answer
and the uploaded document in the answer's source panel.

## Layers

- `pages/workspace_page.py`: browser locators and actions.
- `tests/ui/conftest.py`: page setup and failure artifact attachments.
- `assertions.py`: typed UI checks and the shared required-fact contract.
- `tests/ui/test_workspace_chat.py`: scenario orchestration and Allure steps.

The Page Object targets the pinned AnythingLLM 1.16.2 UI. Accessible labels and
placeholders are preferred. Upstream provides no message test IDs, so the chat
history and final Markdown block use observed DOM classes. The answer locator
excludes the separate Thoughts panel. A source button signals completed generation;
checks do not rely on fixed sleeps or retrying a failed model answer.

## Install and run

From `automation`, with Python 3.12 and the application running:

```bash
python -m pip install -r requirements-ui.lock -r requirements-reporting.lock
python -m pip install --no-deps -e .
python -m playwright install chromium --no-remove
python -m pytest tests/ui --run-ui --browser chromium \
  --tracing retain-on-failure --screenshot only-on-failure \
  --output reports/step-16/browser-artifacts \
  --alluredir reports/step-16/allure-results \
  --junitxml reports/step-16/ui.xml
```

Use a fresh report directory for each run. Browser scenarios are skipped unless
`--run-ui` is supplied; the UI extra is optional for ordinary API/unit runs.
Each test gets a fresh browser context and a uniquely named API workspace.
Existing fixtures delete the document folder and workspace even if UI checks fail.
The developer API key stays in the API layer and is not passed to the browser.

This scenario uses the configured generation model (currently Qwen 3.5 4B), one
answer generation and no judge calls. `--rag-model` only affects RAG-marked tests;
it does not expand UI scenarios into a model matrix. The configured LLM timeout
bounds waiting for the completed answer.

## Evidence

Allure includes the displayed final answer and a screenshot of the answer with its
source panel. On failure, the Playwright plugin retains screenshots and `trace.zip`;
the UI teardown hook attaches them to Allure. Browser artifacts stay in ignored
local report directories because traces may contain application data.

```bash
npm --prefix tools/allure exec -- allure generate reports/step-16/allure-results \
  --output reports/step-16/allure-report
python -m playwright show-trace reports/step-16/browser-artifacts/<test-directory>/trace.zip
```

This browser smoke scenario checks rendering, interaction and required policy
facts. It does not measure faithfulness; the separate live quality scenario does.
Further UI coverage can check document upload, a missing-information answer and
thread isolation, reusing Page Objects and API setup.

## Validation

- One real Chromium UI scenario passed in 33.51 seconds.
- 75 unit checks and six API checks passed without extra answer generation.
- The UI scenario skipped successfully with the Playwright plugin disabled.
- A temporary deliberate failure produced both a screenshot and a trace attached
  to Allure; the diagnostic probe was removed after verification.
- API fixture teardown completed successfully and removed test data.

## Official references

- [Playwright Python installation](https://playwright.dev/python/docs/intro)
- [Locator recommendations](https://playwright.dev/python/docs/locators)
- [Pytest browser artifacts](https://playwright.dev/python/docs/test-runners)
