# Step 14: Combined Quality Report in Allure

## Independent dimensions

The saved paid-leave answer is checked against a versioned profile in
`automation/test_data/quality-paid-leave.json`. A quality summary combines:

1. Required answer facts: paid-leave allowance and notice period.
2. Document sources: expected document identity and required source passages.
3. Faithfulness: an existing completed RAGAS evaluation, bound to the sample's
   exact file checksum.

Each dimension remains separate. Deterministic checks use `passed`/`failed`;
invalid or missing evidence uses `error`. Faithfulness uses `measured`, with no
automatic quality threshold. The overall summary is `checks_passed` only when
deterministic checks pass and judge evidence is valid. This does not imply that
every possible answer-quality requirement has passed. A low faithfulness score
is visible as a measurement, not silently promoted to an approved quality gate.

Expected source identity comes from the actual context's `sourceDocument`
metadata, matched to the profile's policy filename pattern. It is then compared
with the saved response citations; citations do not define their own expectation.
This source adapter targets the pinned AnythingLLM context format. Citations
and actual model contexts remain separate attachments.

## Layers

`assert_required_facts` is shared by live RAG checks and saved-answer reporting.
All assertions remain in `assertions.py`. `reporting.quality` builds the summary
and retains all dimension outcomes. `reporting.allure_report` presents steps and
attachments. The opt-in pytest quality test writes a local JSON summary, renders
every step even after an assertion failure and finally checks the report result.

This step reuses saved application/judge evidence. It makes no model calls,
does not regenerate an application answer and does not require RAGAS imports
to read an existing report. Original judge timestamps/settings are retained;
the report creation time is a separate timestamp. Checksums prevent accidentally
mixing samples, not malicious tampering with all local artifacts.

## Install tooling

From `automation`:

```bash
.venv/bin/python -m pip install -r requirements-reporting.lock -e '.[reporting]'
npm --prefix tools/allure ci
```

Versions: allure-pytest 2.16.2 and Allure Report 3.19.1. The report generator is
project-local and pinned by its npm lockfile; there is no global installation.
The verified environment uses Node.js 24.12.0. Report 3 supports the official
pytest integration; Java is not needed for this generator.

## Generate a report

Choose a captured paid-leave sample from step 10 and its completed faithfulness
report from step 11. Run from `automation`:

```bash
.venv/bin/python -m pytest tests -m 'api or smoke or quality' \
  --quality-sample reports/rag-samples/<capture-id>.json \
  --faithfulness-report reports/faithfulness-<capture-id>.json \
  --alluredir=reports/quality-run/allure-results \
  --clean-alluredir --allure-no-capture

tools/allure/node_modules/.bin/allure generate reports/quality-run/allure-results \
  --output reports/quality-run/allure-report --report-name 'LLM RAG Quality'

.venv/bin/python -m http.server 5055 --bind 127.0.0.1 \
  --directory reports/quality-run/allure-report
```

Open `http://127.0.0.1:5055` while the server is running. Stop the local server
with Ctrl+C in its terminal. Use `tests/test_quality.py` instead of `tests` and
omit `-m` to produce only the offline quality test. Explicitly naming the test
path lets pytest load the custom reporting options before parsing file arguments.

Without both quality arguments, the quality test skips during ordinary test
runs. Supplying only one argument fails clearly. Allure results require
`--alluredir`; ordinary tests can continue using JUnit XML. Use a separate report
directory per run; `--clean-alluredir` removes previous result files from the
selected directory and avoids mixing independent runs.

## What to inspect

Expand the `RAG quality: paid_leave` result to inspect the three steps. Attachments
include the question, final answer, actual contexts, response sources, run
metadata, quality summary and original judge evidence. JSON summaries are also
saved separately under `automation/reports/quality/` with unique filenames.

Automatic stdout/stderr capture is disabled in the documented command; only
selected diagnostic artifacts are attached. Report files contain test policy
data and model prompts, are ignored by Git and are not published automatically.
Serving is restricted to localhost. History/trends and CI hosting are future
steps, not inferred from this single report.

## Verified result

On 2026-10-01, 72 framework unit checks and one offline quality check passed.
A separate report contains 7 API/smoke checks and the quality check: 8 passed.
The generated Allure UI was opened and verified locally. The quality result
contains all three steps and seven top-level diagnostic attachments, with
faithfulness 1.0 recorded from the previous judge run. No model requests were
made during this reporting step.

Unit coverage verifies missing required facts despite faithfulness 1.0, wrong
source identity, mismatched sample checksums, incomplete judge results, scores
inconsistent with verdicts and preservation of later Allure steps after failure.

## Primary references

- [Allure pytest integration](https://allurereport.org/docs/pytest/)
- [Allure Report 3 installation](https://allurereport.org/docs/v3/install/)
- [Compatibility with existing integrations](https://allurereport.org/docs/v3/migrate/)
