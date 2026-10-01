# Step 17: Three additional core UI scenarios

The UI suite now contains four independent scenarios, each using an isolated
browser context, temporary API workspace and indexed English policy document.
Page Objects own actions and locators; the shared Assertions module owns checks.
No extra abstraction or model matrix is added.

| Scenario | Contract |
| --- | --- |
| Paid-leave answer | Question is displayed, final answer contains required facts, uploaded document appears in Sources. |
| Missing gym policy | Final answer acknowledges missing information, addresses gym reimbursement and does not invent an amount. |
| Source details | Opening the cited document displays its expected heading and passages containing both policy facts. |
| History after reload | The same conversation URL, question and exact completed answer remain, with one user question and one assistant message. |

The missing-information check is shared with API tests through
`assert_missing_policy_answer`. Browser checks read the final Markdown answer,
excluding Thoughts. The source-detail locator scopes the upstream overlay by its
heading because AnythingLLM 1.16.2 does not provide a dialog role for that overlay.
The full source title is checked in the Sources list; the overlay heading uses
the UI's 45-character shortening. Question matching accepts the smart apostrophe
introduced by Markdown rendering without weakening the remaining question text.

## Run

Use the UI/reporting installation described in [step 16](step-16-ui-chat.md).
From `automation`:

```bash
python -m pytest tests/ui --run-ui --browser chromium \
  --tracing retain-on-failure --screenshot only-on-failure \
  --output reports/step-17/browser-artifacts \
  --alluredir reports/step-17/allure-results \
  --junitxml reports/step-17/ui.xml
```

Select a single scenario while learning or debugging:

```bash
python -m pytest tests/ui --run-ui -k source_details --alluredir reports/ui-source-check
```

A full run makes four sequential answer generations on the configured model
(currently Qwen 3.5 4B). Reloading history makes no additional generation request.
There are no judge calls, automatic answer retries or parallel model requests.
All four scenarios are skipped by default; API fixture teardown cleans up data
on successful and failed checks. Screenshot/trace retention follows the existing
Playwright options and Allure attachments.

Use a fresh report directory for each run. These checks illustrate browser
interaction and persistence; they do not estimate overall model quality.

## Validation

- 75 unit checks passed after extracting the shared API/UI missing-information check.
- The existing answer scenario and the new history scenario passed in the first
  browser run. Two locator issues (smart apostrophes and shortened source headings)
  were corrected; both affected scenarios passed in a focused rerun (58.81 seconds).
- All four UI contracts therefore passed across the initial and focused runs; the
  original failed-run diagnostics remain locally available.
- All four scenarios skipped with the Playwright plugin disabled, and fixture
  teardown completed successfully for every browser scenario.
