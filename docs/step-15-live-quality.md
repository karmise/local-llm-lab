# Step 15: Live RAG-to-Allure Scenario

## Flow

One explicit pytest scenario creates a temporary workspace, uploads and indexes
the fictional policy, asks the paid-leave question, captures actual model
messages, runs a local faithfulness judge and renders the combined Allure report.
Existing resource fixtures delete the document folder and workspace and verify
their absence during teardown, including after failed checks.

Generation uses one selected model. The judge runs at most two additional calls
with thinking disabled and no retries. Default model selection for this scenario
is Qwen3.5 4B, independent of the ordinary two-model RAG matrix. Multiple models
or repetitions are rejected before fixture setup. This scenario is skipped
unless explicitly enabled; ordinary API and unit runs remain lightweight.

The new test composes existing layers rather than sending HTTP requests itself.
The reusable `evaluate_sample_report` service preserves judge evidence for both
success and error and is also used by the standalone faithfulness CLI. The
sample is checksum-bound to its new judge report. Fact and source checks remain
independent of faithfulness, which is still measured without a quality threshold.

## Run

Install the evaluation and reporting extras as described in steps 11 and 14.
With Docker Desktop and native Ollama running, enable observation from the
repository root:

```bash
mkdir -p .runtime/ollama-capture
docker compose -f compose.yaml -f compose.capture.yaml up -d anythingllm
```

Wait for the application to become available. From `automation`:

```bash
.venv/bin/python -m pytest tests/test_live_quality.py \
  --run-live-quality --capture-rag \
  --rag-model qwen3.5:4b --judge-model qwen3.5:4b \
  --alluredir=reports/live-run/allure-results \
  --clean-alluredir --allure-no-capture \
  --junitxml=reports/live-run/results.xml -v

tools/allure/node_modules/.bin/allure generate reports/live-run/allure-results \
  --output reports/live-run/allure-report --report-name 'LLM RAG Live Quality'

.venv/bin/python -m http.server 5056 --bind 127.0.0.1 \
  --directory reports/live-run/allure-report
```

Open `http://127.0.0.1:5056` while the server runs. Stop it with Ctrl+C.
Use a separate output directory for each run. Invoke the individual test file
to avoid selecting other generation scenarios accidentally.

After testing, restore the ordinary application configuration from the root:

```bash
docker compose -f compose.yaml up -d anythingllm
```

The test requires capture and an Allure results directory before setup. The
Compose overlay is infrastructure setup, not managed automatically by pytest;
enabling it recreates the container while preserving persistent application data.
Running without the active overlay cannot collect an actual request and fails
collection of the sample after the application response.

## Evidence and failures

Captured samples are saved under `automation/reports/rag-samples/`. Each live run
adds `faithfulness.json` and `quality.json` under a unique
`automation/reports/live-quality/<capture-id>/` directory. The sample includes
the temporary workspace slug for traceability. All files remain local and
ignored; they contain no API authorization header.

Allure includes generation/capture and judge-execution steps followed by the
three quality dimensions, with the new sample, judge report and diagnostic
attachments. A judge failure remains an error in its evidence and quality
dimension; fact/source results are retained. Incomplete generation or capture
failure stops before judgement, while resource teardown still runs. Raw capture
may remain available for diagnosis even when no completed sample can be built.

Cleanup failures are test teardown errors, not successful cleanup. Document
cleanup failure does not prevent the outer workspace fixture from attempting
its own deletion. Persistent local evidence is intentionally retained for
review; application test data is deleted.

## Verified run and next coverage

On 2026-10-01, one Qwen3.5 4B live scenario passed in 38.42 seconds, including
setup, generation, two judge calls, report construction and verified teardown.
The Allure report contains successful generation/capture and judge steps plus
all quality dimensions. The application was restored to its ordinary configuration.

All 75 framework unit checks passed. Additional checks cover cleanup after a
failed quality check, continued workspace cleanup when document cleanup fails,
and preservation of judge error evidence with transport closure. Missing capture
and multiple-run requests were rejected before fixture setup. The default live
test invocation was confirmed to skip without resource creation.

Next: UI smoke coverage through Playwright, with Page Objects and existing API
fixtures for data setup. Start with opening a workspace, submitting a question
and displaying the final answer and document sources. Keep UI interaction checks
separate from semantic answer judgement and attach screenshots/traces on failure.
