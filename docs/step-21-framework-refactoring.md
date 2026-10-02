# Step 21: Framework refactoring

The framework now keeps pytest orchestration in explicit modules while preserving
existing integration-test paths, fixture names, CLI flags and Allure display names.
See [Automation](../automation/README.md) for setup and layer-specific commands.

## Findings addressed

| Finding | Change |
| --- | --- |
| `tests/conftest.py` mixed option parsing, HTTP setup, lifecycle, model metadata and capture | Thin plugin registration; separate `options`, `environment`, `resources` and `rag` modules |
| Validation could fail before teardown was registered | Register resource finalizers after receiving usable creation identifiers, before remaining contract checks |
| Workspace creation only checked the slug prefix | Validate a positive integer ID and the requested name as well |
| UI completion depended on Sources, and `.last` could point to an earlier answer | Target the next persisted reply after send; wait for nonempty content and the visible Send control; assert citations separately |
| Scenario questions, reference answers and facts were duplicated | Shared paid-leave and missing-policy profiles for API, UI and quality flows |
| Live-option checks ran before marker/name deselection | Validate only the selected tests; detect a disabled/missing Playwright plugin before setup |
| UI unnecessarily required Allure | Ordinary UI and browser checks only require the UI dependencies |
| HTTP session ownership was repeated; redirects could hide an API response | Context-manager lifetime and explicit redirects disabled by default; no automatic retries |
| Settings accepted malformed ports | Validate ports and whitespace before making network requests |
| Unit tests verified outputs through framework assertion helpers | Plain pytest assertions with normal diff diagnostics |
| Cleanup test called private fixture wrappers discovered through `sys.modules` | Real pytest setup/call/teardown regression tests including multiple failures |
| Core transport, configuration and collection behavior had little direct coverage | Tests for credentials, timeout/redirect behavior, upload handles, URL encoding, model matrix and opt-in gates |
| CI did not verify formatting or Page Objects | Ruff formatting/import/Bugbear checks; separate offline Chromium job |

Unit tests guard Requests calls and set telemetry options with scoped patches.
Browser regressions use local HTML and delayed DOM updates to reproduce earlier
answer, empty answer, missing source and disabled-send cases without model calls.
The real application keeps Send disabled with an empty composer even after a
completed reply; the regression suite explicitly covers that state.

## Validation

Local validation on 2026-10-01 used the existing Python 3.12.3 environment and
the running AnythingLLM instance:

| Layer | Result |
| --- | --- |
| Original unit suite before refactoring | 78 passed |
| Final unit suite | 140 passed |
| Live health/authentication/workspace/indexing API checks | 7 passed |
| RAG checks across Qwen3.5 4B and Qwen2.5 7B | 4 passed |
| Offline Chromium Page Object regressions | 4 passed |
| Real application Chromium UI scenarios | 4 passed |
| Saved-answer quality report using existing checksum-bound judge evidence | 1 passed |
| Ruff lint and format checks | Passed |
| Installed dependency consistency (`pip check`) | Passed |

The final browser/UI run completed all eight checks in 113.86 seconds.
The initial UI attempt exposed an invalid assumption that Send becomes enabled
after generation. That check was corrected and covered by the disabled-composer
regression before rerunning all browser/UI scenarios. The interrupted test's
workspace and document folder were verified absent. A final API lookup found
zero temporary automation workspaces; successful fixture teardown also verified
the document-folder cleanup for each completed scenario.

With optional pytest plugins disabled, the unit suite passes and all nine
opt-in browser/UI/live-quality checks skip before external fixture setup.
The workflow YAML and documentation links were validated locally; the GitHub
jobs themselves have not been executed as part of this local change.

Fresh live RAGAS/judge evaluation was not run. Its mocked adapter tests pass;
the saved-quality scenario validates existing judge evidence without generating
new answers or scores. No generation retries or acceptance thresholds were added.

Generated evidence is under ignored `automation/reports/refactor/`:
`unit.xml`, `api-browser.xml`, `final-ui.xml`, `saved-quality.xml`, `final-allure/`
and `saved-quality-allure/`. The initial `allure-results/` preserves the four
successful RAG results and the original UI failure for investigation.

## Remaining limits

The [known boundaries](../automation/README.md#known-boundaries) are explicit:
version-dependent upstream DOM/capture contracts, regex-based smoke criteria,
uncalibrated judge scores, uncertain creation after transport failures and
unverified concurrent live runs. The refactor does not change quality thresholds
or add retries to make generated answers pass.

## References

- [pytest fixture finalization and safe teardown](https://docs.pytest.org/en/stable/how-to/fixtures.html#teardown-cleanup-aka-fixture-finalization)
- [pytest hook execution order](https://docs.pytest.org/en/stable/how-to/writing_hook_functions.html#hook-function-ordering-call-example)
- [Playwright retrying assertions](https://playwright.dev/python/docs/test-assertions)
- [Playwright locators](https://playwright.dev/python/docs/locators)
