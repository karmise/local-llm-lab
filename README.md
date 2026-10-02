# LLM / RAG Application Testing

Start with the [framework onboarding guide](docs/framework-onboarding.md) to understand
the architecture and testing decisions in about one hour.

[![Framework checks](https://github.com/karmise/local-llm-lab/actions/workflows/framework-checks.yml/badge.svg)](https://github.com/karmise/local-llm-lab/actions/workflows/framework-checks.yml)

A local AnythingLLM environment and a Python automation framework for testing
LLM and retrieval-augmented generation workflows.

The framework is developed incrementally. It currently provides a layered API
client, authentication checks, workspace lifecycle, document retrieval,
grounded policy-answer and missing-information checks across two generation models,
actual context capture and local RAGAS faithfulness evaluation with hand-labelled
judge controls, four Playwright UI scenarios and automated Ruff/unit checks in
GitHub Actions. Broader model evaluation and integration CI are planned extensions.

## Project structure

```text
local-llm-lab/
├── compose.yaml              # AnythingLLM service
├── lab.sh                    # environment management
├── config/                   # application configuration templates
├── docs/                     # setup and framework guides
├── evidence/                 # environment metadata and verification summaries
└── automation/               # standalone Python project
    ├── pyproject.toml
    ├── requirements.lock
    ├── src/llm_testkit/       # settings, HTTP transport and API clients
    ├── tests/
    ├── test_data/            # fictional policy documents
    ├── reports/              # local reports; excluded from Git
    └── .venv/                # local environment; excluded from Git
```

## Start and stop

From the project root:

```bash
./lab.sh start
./lab.sh status
./lab.sh stop
```

Open http://localhost:3001. `start` starts Docker Desktop, the native Ollama
service and AnythingLLM, then opens the browser once the application is ready.
Ollama runs as a user Homebrew service with login autostart. Docker Desktop must
remain running while AnythingLLM is in use.

```bash
./lab.sh restart
./lab.sh logs
./lab.sh stop-ollama
```

`stop` stops AnythingLLM and preserves its data. `stop-ollama` stops Ollama and
disables its service autostart; the next `start` enables it again.
To unload models, use `ollama stop qwen3.5:4b`, `ollama stop qwen2.5:7b` and
`ollama stop bge-m3:567m`.

## Run the tests

See [step 1: Python setup](docs/step-01-python-project.md) and
[step 2: layered API framework](docs/step-02-base-layer.md).
Continue with [step 3: developer API access](docs/step-03-developer-api.md).
Then review [step 4: temporary workspace lifecycle](docs/step-04-workspace-lifecycle.md).
Response validation is centralized in the [assertions module](docs/assertions.md).
See [step 5: document indexing and retrieval](docs/step-05-document-indexing.md).
Continue with [step 6: grounded policy answer](docs/step-06-rag-answer.md).
See [step 7: missing policy information](docs/step-07-missing-information.md).
See [step 8: model comparison](docs/step-08-model-comparison.md) for parameterized
RAG runs and result metadata.
See [step 9: repeated-run stability](docs/step-09-stability.md) for independent
repetitions and an offline outcome summary.
See [step 10: actual model context capture](docs/step-10-context-capture.md) for
opt-in collection of evaluation inputs from the Ollama SDK boundary.
Continue with [step 11: local RAGAS faithfulness](docs/step-11-faithfulness.md)
to evaluate a saved sample separately from ordinary test runs.
See [step 12: hand-labelled judge controls](docs/step-12-judge-controls.md)
for supported, contradicted and invented-claim calibration checks.
Continue with [step 13: expanded judge controls](docs/step-13-expanded-judge-controls.md)
for paraphrases, incomplete answers and working/calendar-day substitutions.
See [step 14: combined quality report in Allure](docs/step-14-allure-quality-report.md)
for separate fact/source checks and saved faithfulness evidence with diagnostic attachments.
Continue with [step 15: live RAG-to-Allure scenario](docs/step-15-live-quality.md)
for an explicit single-model run from test-data setup through verified cleanup.
See [step 16: workspace UI chat](docs/step-16-ui-chat.md) for an opt-in Playwright
scenario with Page Objects, shared assertions and browser failure evidence.
Continue with [step 17: core UI scenarios](docs/step-17-ui-core-scenarios.md) for
missing information, source details and history persistence.
See [step 18: GitHub Actions](docs/step-18-github-actions.md) for automatic
Ruff checks, offline unit tests and downloadable CI reports.
See [step 19: reporting layers](docs/step-19-reporting-layers.md) for reusable UI
steps, selective API steps and tests without direct Allure integration.
See [step 20: readable test titles](docs/step-20-test-titles.md) for explicit
English display names and distinguishable parameter variants in Allure.
See [step 21: framework refactoring](docs/step-21-framework-refactoring.md) for
modular fixtures, failure-safe cleanup and browser regression coverage.
The [Automation guide](automation/README.md) lists commands for each test layer;
use `python -m pytest tests/unit -q` for checks without running services.

```bash
cd automation
source .venv/bin/activate
python -m pytest -v
```

In PyCharm, open the project root or `automation`. Select the existing
`automation/.venv/bin/python` interpreter and use `automation` as the test
working directory.

See [Git workflow](docs/git-workflow.md) for the first commit and subsequent
feature branches. The Git repository root is `local-llm-lab`.

## Components

| Component | Purpose |
| --- | --- |
| AnythingLLM | Web UI, API, document processing and prompt assembly; Docker |
| Ollama | Native model server using Apple Silicon |
| Qwen3.5 4B | Default generation model |
| Qwen2.5 7B | Baseline generation model for comparisons |
| BGE-M3 567M | Document and question embeddings |
| LanceDB | Vector storage in the persistent Docker volume |

Request flow: question → question embedding → document retrieval → prompt with
retrieved context → model response → sources returned by AnythingLLM.
Evaluate model responses against facts and supporting sources rather than exact
string matches.

## Switch generation models

```bash
./lab.sh model qwen3.5:4b
./lab.sh model qwen2.5:7b
```

This changes the generation model for `Company Policy Lab`. Refresh the browser
page afterwards. Both models remain installed. The global default is Qwen3.5 4B.
BGE-M3 and the index remain unchanged, so switching generation models does not
require re-embedding documents. The helper currently uses an internal UI API;
the framework will use the documented developer API for subsequent operations.

For comparisons, record the model digest, prompt, temperature, thinking mode and
question set. A failure that occurs consistently with one model indicates a
model-dependent result relative to the acceptance criteria. A flaky test changes
its result across repeated runs with the same configuration.

## Data and configuration

- Application documents, chat history, SQLite and vectors: Docker volume
  `denis-llm-lab_anythingllm-storage`.
- Runtime configuration and secrets: `.runtime/anythingllm.env`, excluded from Git.
- Ollama models: `~/.ollama/models`.
- Configuration templates: `config/anythingllm.env.example` and `config/workspace.json`.
- Fictional English policy: `automation/test_data/company-policy.txt`.
- Environment versions and verification summaries: `evidence/`.

The English policy and workspace configuration are templates. Editing these files
does not automatically replace documents or settings in an existing workspace.
Previously indexed documents and chat history remain in the Docker volume. Apply
the English workspace configuration and replace/re-index the policy document
before evaluating the English corpus; avoid retaining both language versions in
the same comparison workspace.

Stopping, restarting and `docker compose down` without `-v` preserve the volume.
`docker compose down -v` deletes application data. Keep runtime secrets, database
exports and logs containing real data out of Git. The application is bound to
`127.0.0.1` for local access.

## Manual RAG checks

Use the `Company Policy Lab` workspace. Upload the policy and save/embed it into
the workspace. Attaching a file to a message is a separate workflow; RAG checks
require an indexed document.

| Question | Expected facts |
| --- | --- |
| How much paid leave is available per year, and when must a request be submitted? | 23 working days; at least 12 calendar days before leave starts |
| What are the daily travel allowance and hotel reimbursement limit? | KGS 2700 per calendar day; KGS 8400 per night with a receipt |
| How much leave can be carried over, and when must it be used? | Up to 6 working days; by March 31 of the following year |
| How does the company reimburse gym memberships? | The document contains no such policy; the model must not invent an amount |

The first three answers should cite relevant policy passages. After restarting,
the workspace, indexed document and history should remain available.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Application unavailable | Docker Desktop; `./lab.sh status`; `./lab.sh logs` |
| Model connection failure | `curl http://127.0.0.1:11434/api/tags`; container Ollama URL `http://host.docker.internal:11434` |
| Missing model | `ollama pull qwen3.5:4b`, `ollama pull qwen2.5:7b` or `ollama pull bge-m3:567m` |
| Document not retrieved | Workspace indexing, embedding provider and returned sources |
| Slow response | Initial model loading; `ollama ps`; available memory |
| Port 3001 unavailable | Check its owner with `lsof -iTCP:3001 -sTCP:LISTEN` before changing services |

Inside the container, `localhost` refers to the container itself. Use
`host.docker.internal` to reach native Ollama. Changing the embedding model
requires re-indexing: vectors from different models are not interchangeable.

## Planned coverage

Additional question/answer checks; larger stability experiments;
UI coverage; reference question sets; RAG evaluation and CI quality gates.

Use opt-in context capture before adding RAGAS. Returned source references alone
do not establish the complete model context. A successful answer
is an integration smoke result, not a statistical estimate of model accuracy.

## Official references

- [AnythingLLM Docker setup](https://docs.anythingllm.com/installation-docker/local-docker)
- [AnythingLLM Ollama troubleshooting](https://docs.anythingllm.com/ollama-connection-troubleshooting)
- [Ollama on macOS](https://docs.ollama.com/macos)
- [Qwen2.5 7B](https://ollama.com/library/qwen2.5:7b)
- [Qwen3.5 4B](https://ollama.com/library/qwen3.5:4b)
- [BGE-M3](https://ollama.com/library/bge-m3)
