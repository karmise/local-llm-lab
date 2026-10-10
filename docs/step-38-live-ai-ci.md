# Fresh AI evidence and commit-bound CI gates

## What this stage connects

`Fresh RAG quality` starts a disposable AnythingLLM/Ollama environment on an
Ubuntu GitHub-hosted runner, generates new answers through the real application,
captures the actual model context and evaluates it using RAGAS and a fixed
Qwen3.5 4B judge. A separate job downloads the same-run evidence and applies
the existing assertions and experimental score gates with **zero model calls**.

This manual workflow complements automatic framework checks. It does not reuse
the October 8 local benchmark as if it were evidence for a later commit.
The existing single-sample `Saved RAG quality gate` remains available separately.

## Reviewed scope and workload

| Profile | Models | Fresh answers | Maximum generation/judge calls |
| --- | --- | ---: | ---: |
| `smoke` | `primary` | 2 | 19 |
| `smoke` | `comparison` | 4 | 32 |
| `curated` | `primary` | 4 | 43 |
| `curated` | `comparison` | 8 | 80 |

`curated` is the default: paid leave, approver, receipt boundary and unsupported
gym benefits. `smoke` includes paid leave and gym benefits only. `primary` uses
Qwen3.5 4B; `comparison` adds Qwen2.5 7B. Both use the same Qwen3.5 judge.
Three narrow labelled faithfulness controls consume six calls in either preset.
Embedding/indexing calls are outside the generation/judge budget.

Execution is serial, with one answer per case/model and no retries. All
applicable dimensions must pass; refusal metrics remain N/A, not perfect scores.
Missing cells, invalid evidence, judge-control mismatches, timeouts and resource
cleanup failures cannot become a passing aggregate. This curated slice does not
claim complete coverage of the 16-case golden dataset or other test matrices.

The hosted runner uses CPU inference. It is slower than the local Apple Silicon
environment, and timing results from the two environments are not equivalent.
Each API generation may take up to 600 seconds; judge calls use the same timeout.
The benchmark step has an 80-minute limit inside a 120-minute producer job,
leaving time for failure artifacts and cleanup. A timeout is a failure, not a
reason to silently repeat an answer. Downloads have their own 30-minute limit.

## Isolated application setup

`compose.ci.yaml` is a standalone configuration, never a local Compose overlay.
It pins AnythingLLM 1.16.2 and Ollama 0.40.2 by image digest, puts the two services
on an internal Compose network and binds their published ports to loopback.
Each run/attempt has a unique `rag-ci-...` project and its own disposable volumes.
The final cleanup removes only those volumes; it never touches the local lab.

`config/ci-models.json` records the reviewed generation and BGE-M3 embedding
digests. Downloaded tags must match before generation starts. A changed upstream
tag fails preflight and needs an explicit reviewed lock update. Runtime version
and weights are retained in `environment.json`.

The fresh single-user application creates a disposable API key through its
installed management endpoint. Bootstrap is restricted to GitHub-hosted runners,
refuses an existing key, masks the new secret in Actions, verifies authentication
and writes an owner-only file. No local API key, repository secret or external
model account is required. The private runtime directory and storage database are
excluded from uploads. This setup is specifically for the disposable pinned
application, not a way to administer an existing production application.

## Evidence validation and reporting

The producer saves the plan before model calls. Benchmark artifacts include:

- exact dataset, source policy, gate and control baselines;
- captured requests/contexts, answers, generation JUnit and cleanup outcomes;
- all raw judge calls and independently validated metric verdicts;
- model/configuration identity, answer durations and the full declared matrix;
- commit/source identity, aggregate Markdown/JSON and pending review diagnostics.

`ci.benchmark.validate_ci_benchmark` compares those baselines byte-for-byte with
the tested checkout, verifies the complete requested preset and binds captured
samples to that framework's source checksum and actual golden test definition.
It rejects substituted revisions, changed expectations, incomplete plans and
unreviewed generation/judge weights. It then reuses `load_saved_benchmark` to
recompute every applicable acceptance outcome from original evidence.

The saved pytest scenario obtains prepared evidence from a fixture and calls
the shared reporting/assertion layer. There is no branching, model setup or raw
assertion logic inside the test. Allure retains every case and the final outcome.
HTML generation and uploads also run after a failed quality test.

Artifacts are retained for 14 days:

- `fresh-rag-evidence`: plan, environment, benchmark and service diagnostics;
- `fresh-rag-gate-reports`: JUnit, Allure results and generated Allure HTML.

Download them before expiry to retain evidence longer. Public fictional policy
and model-generated text are intentional report contents; credentials are not.
Early setup failures can have diagnostics without a completed benchmark. Those
runs fail the gate. Hard cancellation can leave a pending status and partial or
absent artifacts; pending never authorizes merging.

## Run for the exact commit

In GitHub, select **Actions → Fresh RAG quality → Run workflow**, leave the
workflow branch as `main` and enter the full target commit SHA. An empty SHA
tests the dispatched main commit. For a feature branch, pass its latest full SHA.

From the repository root:

```bash
gh workflow run rag-live-quality.yml --ref main \
  -f revision="$(git rev-parse HEAD)" -f profile=curated -f models=primary
```

The trusted main workflow resolves that exact commit before checkout. Producer
and gate jobs have read-only repository permissions. Separate status-writing
jobs never check out or execute the target code. The context starts pending and
becomes success **only when both the producer and independent gate succeed**.
Failed, missing, skipped or cancelled required work cannot publish success.
The workflow is dispatchable only from main. It has no PR trigger and installs
no runner on a developer machine.

## Merge policy

A manual workflow job check alone does not satisfy a PR's required status check.
This pipeline publishes a commit status called `RAG benchmark quality` for
`curated` runs. A `smoke` run publishes **a different context**, `RAG smoke quality`,
so a smaller passing matrix cannot satisfy the curated merge requirement.
Statuses target the tested SHA, not the workflow's main SHA. Any subsequent code
commit requires new evidence. Concurrent runs for the same target are serialized.

Repository branch protection should require these contexts on `main`:

- `Ruff and unit tests`;
- `Offline Page Object checks`;
- `RAG benchmark quality`.

The two automatic job checks use the GitHub Actions app as their expected source.
The AI context is a commit status written using the workflow's GitHub token.
The branch should require being current with main and disallow force pushes and
deletion. Administrator bypass remains available for deliberate repository
maintenance; a direct administrator push is not evidence that quality passed.
The repository setting is operational state, not implied by this YAML. Inspect
the branch protection API to confirm its current required contexts.

The semantic thresholds remain **experimental**. The gate demonstrates an
enforced engineering acceptance mechanism, not calibrated release suitability,
clinical accuracy or GxP approval. Disputed scores remain failed until reviewed;
neither workflow nor report builder changes thresholds to obtain a green run.

## Verification

Offline verification passed 585 unit cases with 81.48% combined statement/branch
coverage, above the 75% floor. New checks cover all four workload presets,
commit/source/baseline/plan/model provenance, preserved generation errors,
bootstrap refusal on developer machines, private key creation and weight checks.
Ruff, isort, YAPF, actionlint 1.7.12 and Compose configuration validation passed.

Actual hosted execution and repository protection are verified separately after
publication; a local unit pass must not be reported as a passing live AI run.

## References

- [GitHub: which workflow checks satisfy required PR checks](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks).
- [GitHub: commit statuses reflected in pull requests](https://docs.github.com/en/rest/commits/statuses).
- [GitHub: protected branches and administrator bypass](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
- [Ollama: Docker and memory/concurrency configuration](https://github.com/ollama/ollama/blob/main/docs/faq.mdx).
- [AnythingLLM 1.16.2 API-key management implementation](https://github.com/Mintplex-Labs/anything-llm/blob/v1.16.2/server/endpoints/system.js).
