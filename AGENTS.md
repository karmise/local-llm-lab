# AGENTS.md

Context for AI coding agents and new contributors. Read this first, then
[docs/decisions.md](docs/decisions.md) before changing behaviour.

## What this project is

A local LLM/RAG lab and the Python framework that tests it. AnythingLLM 1.16.2
(Docker) handles documents, retrieval and chat; native Ollama serves the
generation models (`qwen3.5:4b` default, `qwen2.5:7b` for comparison) and BGE-M3
embeddings. The corpus is one short fictional company policy, kept small on
purpose: the product is the evaluation framework, `automation/src/llm_testkit`,
not the corpus.

The framework checks the application at several layers: API, RAG answers,
golden dataset, UI, conversation, prompt/adversarial/bias regression and
performance. It evaluates captured answers with RAGAS metrics and a local judge
that is calibrated against hand-labelled controls, and applies experimental
quality gates. A manual GitHub workflow produces fresh evidence per commit and
publishes the required merge status `RAG benchmark quality`.

## Layout

```text
README.md                    setup, lab.sh, components, manual checks
compose*.yaml, lab.sh        local services; compose.capture.yaml adds context capture
config/                      workspace templates (benchmark: temperature 0)
docs/                        onboarding, evaluation guide, decisions, test plan
automation/
  src/llm_testkit/           the framework
    core/                    HttpClient transport and shared helpers
    clients/                 BaseClient → AnythingLLMClient (.workspaces, .documents), OllamaClient
    pages/                   BasePage → WorkspacePage
    assertions/              acceptance checks: fields, api, answers, quality, ui
    evaluation/              RAGAS metrics, judge controls, benchmark runner
    reporting/               Allure, gates, drift, stability, CI evidence
    qualification/           IQ/OQ/PQ plan and packaging
    pytest_plugin/           the framework's pytest plugin: CLI options, opt-in selection, evidence labels
  src/test_support/          what tests use: fixtures (live resources) and builders (unit data)
  tests/                     grouped by level, then by feature:
    unit/                    offline framework tests (~2000, plain pytest), mirrors src/llm_testkit
    browser/                 Page Object checks on local HTML
    environment/ api/        installation and API, no generation
    rag/ regression/ ui/     generated answers; regression and UI are opt-in
    quality/ performance/    evidence, gates and load
  test_data/                 policy, golden dataset, catalogs, gates, plan
```

## Commands

Run from `automation` with `.venv` activated (Python 3.12).

```bash
python -m pytest tests/unit -q          # offline; bare `pytest` also runs live tests
python -m ruff check src tests
python -m isort --check-only --diff src tests
python -m yapf --diff --recursive src tests
```

Format with `isort` and `yapf --in-place`, never `ruff format`; the line limit
is 120. Opt-in suites and their commands are in
[docs/evaluation-guide.md](docs/evaluation-guide.md). Live tests need AnythingLLM
running (`./lab.sh start`) and a developer API key.

## Rules

- **No retries, no relaxed criteria.** Do not add retries, widen regexes to fit a
  model's latest output or change a threshold to turn a run green. Fix the cause,
  and record a deliberate change with a version bump and its rationale.
- **Reference data is checksum-bound.** Editing `golden-policy.json`, catalogs,
  `quality-gates.json` or `qualification-plan.json` requires bumping their
  version and updating the dependent checksums (`golden_dataset_sha256`,
  `data_sha256`); unit tests fail otherwise.
- **Validation raises.** Framework code raises `ValueError`, not `assert`, which
  `python -O` removes. Incomplete or malformed evidence must be an error.
- **Unit tests** use plain `assert` and `pytest.raises(..., match=...)`, never the
  application `assertions` package. Each rejection has a paired acceptance case.
  Shared data comes from `test_support/builders`; tests stay linear, with no
  branches.
- **Live fixtures** register cleanup right after creation and delete only
  resources they created.
- **The repository is public.** Never commit secrets, `.runtime/`, reports or
  personal notes; `reports/` and `.runtime/` are git-ignored.
- **Dependencies** are pinned per profile in `requirements-*.lock`. Do not add
  one-off tools such as mutmut to them.

## Workflow

- Branch from `main`; commit messages use conventional prefixes (`feat:`,
  `fix:`, `docs:`, `refactor:`, `test:`, `ci:`, `style:`).
- `main` requires `Ruff and unit tests`, `Offline Page Object checks` and
  `RAG benchmark quality`. The last is set by the manual workflow:
  `gh workflow run rag-live-quality.yml --ref main -f pull-request=<n> -f profile=curated -f models=primary`.
  It takes about two hours, and every new commit needs a new run.
- Squash-merge pull requests.
