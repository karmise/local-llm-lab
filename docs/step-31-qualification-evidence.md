# Requirement traceability and educational IQ/OQ/PQ evidence

The [test plan](test-plan.md) describes intended use, risk-based scope, strategy,
criteria, evidence and change control. The machine-readable plan maps each
requirement to real test definitions and explicitly declares parameter matrices.
Pytest validates it at collection and attaches requirement/phase labels to JUnit
and optional Allure. Tests continue to use the existing fixtures and Assertions;
qualification does not duplicate test logic.

## Protocol scope

IQ verifies the supported runtime, service health and installed generation models.
OQ verifies authentication, resource lifecycle/cleanup and document indexing/search.
PQ links representative intended-use checks: golden answers, prompt changes,
adversarial inputs, paired bias cases, UI, saved quality gates and bounded latency.

These are educational software protocols, with review pending. The terminology
is informed by [EU GMP Annex 15, sections 3.8–3.14](https://health.ec.europa.eu/document/download/7c6c5b3c-4902-46ea-b7ab-7608682fb68d_en?filename=2015-10_annex15.pdf).
That document concerns qualification and validation in regulated manufacturing;
this local fictional-policy project does not establish GxP compliance. Formal
approved protocols, qualified infrastructure, controlled records, electronic
signatures and clinical acceptance criteria are outside this implementation.

## IQ and OQ execution

Prerequisites: application/Ollama online, both models installed, developer key
configured, Python 3.12 environment and base locks installed. Review the plan's
version, criteria and data hashes before execution. Run from `automation`:

```bash
.venv/bin/python -m pytest tests/test_installation.py tests/test_health.py -v \
  --junitxml reports/qualification/iq.xml
.venv/bin/python -m pytest tests/test_authentication.py tests/test_documents.py \
  tests/test_workspace.py -k 'not configured_workspace' -v \
  --junitxml reports/qualification/oq.xml
```

IQ has three checks; OQ has five. These commands make no answer-generation or
judge calls. OQ performs document embedding/indexing. Allure evidence is optional;
install the reporting extra and add a distinct `--alluredir` for each run when
needed. Runtime/model inventory is also retained in JUnit properties.

## PQ execution

Choose budgets and follow the existing guides for each requirement:

| Requirement | Guide | Declared coverage |
| --- | --- | --- |
| REQ-GOLDEN | [Golden dataset](step-22-golden-dataset.md) | 32 answer generations |
| REQ-PROMPT | [Prompt regression](step-25-prompt-regression.md) | 64 answer generations |
| REQ-ATTACK | [Adversarial](step-26-adversarial-inputs.md) | 12 answer generations; capture overlay required for document attacks |
| REQ-BIAS | [Counterfactual checks](step-30-bias.md) | 12 answer generations |
| REQ-UI | [UI scenarios](step-17-ui-core-scenarios.md) | All four declared application tests |
| REQ-QUALITY | [Quality gates](step-27-quality-gates.md) | Saved paid-leave evidence with explicit gates enabled |
| REQ-PERFORMANCE | [Performance](step-29-performance.md) | Separate health and RAG batches with recorded thresholds |

Start with narrow scopes during development. Keep JUnit for each run. The complete
golden/prompt/attack/bias matrices alone require 120 generations, before UI,
performance or fresh evaluation calls. They are not executed by packaging or by
ordinary CI. Full PQ cannot be claimed from the focused examples in these guides.
Performance requirements declare both workloads, but no production concurrency
or latency SLO. Model/iteration metadata remains available for reviewing each run.

## Build and verify a scoped package

Package only phases actually intended for review. IQ/OQ can be packaged independently
of PQ. Choose a new output directory for each package:

```bash
.venv/bin/python -m llm_testkit.qualification.package \
  --phase IQ --phase OQ \
  --junit reports/qualification/iq.xml --junit reports/qualification/oq.xml \
  --output reports/qualification/iq-oq-package
```

Repeat `--junit` for multiple inputs and `--attach reports/path/to/metrics.json`
for explicit auxiliary evidence. Attachments must reside under `automation/reports`;
the tool does not scan runtime storage or include credentials automatically.
Preserve raw files with appropriate access controls; review explicit attachments
before sharing them. Generated reports and packages stay ignored by Git.

The package contains `definitions/` (plan, bound datasets, mapped test source,
framework source and dependency/configuration files within automation), exact
`junit/` inputs, optional `attachments/`, `traceability.json`, `summary.md`, and
`manifest.json` with SHA-256 checksums/sizes. Verify it without application calls:

```bash
.venv/bin/python -c "from pathlib import Path; from llm_testkit.qualification.package import verify_package; verify_package(Path('reports/qualification/iq-oq-package'))"
```

Review traceability and original JUnit/attachments together. Packaging returns
nonzero for failed/incomplete execution, while preserving the package for review.
Existing destinations are never overwritten. A pass means the declared selected
cells passed; review remains pending and the report makes no compliance claim.

## Failure semantics

- Missing matrix cells, skips, setup/teardown errors or stale/unbound plan/source
  metadata produce incomplete evidence. A measured quality report without gates
  cannot satisfy REQ-QUALITY.
- A test failure produces failed evidence. A later successful result cannot erase
  it; selecting a clean run deliberately is a review decision, not an automatic retry.
- Duplicate call/teardown entries are aggregated; cleanup errors remain visible.
  Conflicting metadata cannot close a requirement.
- Legacy JUnit without traceability properties cannot be treated as proof of the
  current plan. Changed code/data requires fresh execution under the new baseline.
- Copies are checked against the parsed input/source baselines before publication;
  changed files or a partial build do not leave a completed output directory.

Checksums protect copied bytes against accidental corruption relative to an
unsigned manifest. They do not authenticate the author, prevent deliberate
rewriting of evidence, or provide an immutable regulatory audit trail.

## Implementation validation

The offline suite passed 402 tests with 80.11% combined statement/branch framework
coverage. Qualification cases exercise partial matrices, stale provenance,
retained failures, teardown entries, conflicting metadata, checksum corruption,
unsafe manifest paths, changed inputs during copying and real pytest/JUnit output.
Ruff and dependency consistency checks passed.

On the local application, three IQ and five OQ checks passed, including verified
cleanup. Their scoped package passed integrity verification for all 78 listed
files; native Allure requirement/phase labels were present. Packaging those same
inputs as PQ deliberately returned incomplete with a nonzero exit code. No full
PQ matrix was run, and no human approval is represented by these results.
