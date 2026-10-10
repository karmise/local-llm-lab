# Quality gates and framework coverage

This stage introduces two separate controls: a code coverage floor for the framework and explicit score thresholds for saved application evidence. Neither control estimates clinical safety or model accuracy over an unseen population.

## Framework CI gate

The push/pull-request workflow runs unit tests through pinned Coverage.py with branch tracking and subprocess instrumentation. The latter includes the fixture/collection tests that launch child pytest processes. All `llm_testkit` modules are included, including uncovered browser and live-generation code; there are no module omissions to inflate the result.

The minimum combined statement/branch coverage is **75%**, configured in `pyproject.toml`. The initial measured baseline informed this floor. Failed tests, lint/format failures, or coverage below the floor fail the CI job. XML, JSON, text and JUnit evidence are uploaded even on failures. The offline browser job remains separate.

From `automation`:

```bash
export COVERAGE_FILE="$PWD/reports/coverage/.coverage"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m coverage run -m pytest tests/unit
.venv/bin/python -m coverage combine
.venv/bin/python -m coverage xml --fail-under=0 -o reports/coverage/coverage.xml
.venv/bin/python -m coverage report
```

Use a fresh report directory for an independent measurement. Coverage.py is pinned in the `dev` extra and `requirements-dev.lock`. Coverage is about exercised framework code, not the number of golden cases executed or the quality of generated answers. [Coverage configuration reference](https://coverage.readthedocs.io/en/latest/config.html).

## Saved application quality gate

`test_data/quality-gates.json` defines illustrative minimum scores:

| Metric | Minimum |
| --- | --- |
| Faithfulness | 0.90 |
| Factual correctness F1 | 0.80 |
| Context precision | 0.80 |
| Context recall | 0.90 |

The catalog marks these thresholds **experimental**, with a version, rationale and recorded SHA256. They demonstrate an acceptance mechanism; they have not been calibrated on a representative benchmark. Raising, lowering or changing them requires reviewing the rationale and version rather than editing tests to match a failing answer.

Add `--quality-gates test_data/quality-gates.json` to the saved quality command with all four evidence files:

```bash
.venv/bin/python -m pytest tests/test_quality.py \
  --quality-sample reports/evidence/sample.json \
  --faithfulness-report reports/evidence/faithfulness.json \
  --correctness-report reports/evidence/correctness.json \
  --relevance-report reports/evidence/relevance.json \
  --quality-gates test_data/quality-gates.json \
  --alluredir reports/quality-gate/allure-results \
  --junitxml reports/quality-gate/results.xml
```

The current saved-report profile is `paid_leave`. Each evidence file is validated against the sample and current reviewed expectations before threshold checks. Raw-score inconsistencies, wrong checksums, synthetic correctness controls, missing metrics, incomplete evaluation, and low scores fail the test. Facts and sources must still pass: high judge scores cannot override a failed deterministic check. Equality at a threshold passes. Every independent dimension remains visible in Allure, including failures. Without `--quality-gates`, measurements retain their previous exploratory behavior.

## CI consumption of application evidence

`.github/workflows/rag-quality-gate.yml` is both reusable and manually dispatchable. A producer must upload a same-repository artifact containing `sample.json`, `faithfulness.json`, `correctness.json` and `relevance.json`. The reusable workflow consumes the artifact in the same run; manual dispatch accepts a completed producer run ID and artifact name. It applies the exact saved pytest gate and preserves failure evidence. It performs **zero model calls** and does not install evaluation dependencies.

A caller can run the gate after an evidence-producing job:

```yaml
quality:
  needs: produce-evidence
  uses: ./.github/workflows/rag-quality-gate.yml
  with:
    evidence-artifact: quality-evidence
```

The `produce-evidence` name above is an integration example for the single-sample consumer. The later [fresh AI CI stage](step-38-live-ai-ci.md) adds a real manually dispatched producer with disposable hosted AnythingLLM/Ollama services and a separate complete-benchmark consumer. Ordinary push checks still make no model calls. A passing saved paid-leave sample must not be presented as a passing full golden/adversarial matrix.

Failing a workflow blocks its downstream dependent jobs. Requiring its status before a GitHub merge additionally needs repository branch protection/rulesets; this change does not configure repository administration settings.
