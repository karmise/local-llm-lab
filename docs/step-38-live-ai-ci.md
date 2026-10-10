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
It pins AnythingLLM 1.16.2 and Ollama 0.35.0 by image digest, puts the two services
on an internal Compose network and binds their published ports to loopback.
Each run/attempt has a unique `rag-ci-...` project and its own disposable volumes.
The final cleanup removes only those volumes; it never touches the local lab.

`config/ci-models.json` records the reviewed generation and BGE-M3 embedding
digests. `config/ci-model-manifests/` preserves the exact original manifest bytes.
CI downloads their immutable blobs, verifies every size/SHA-256 and installs the
manifests before starting Ollama. It does not resolve mutable tags with `pull`.
The installed API catalog must still match before generation. Runtime version
and weights are retained in `environment.json`. Model updates require explicit
review of the lock and manifest; missing upstream blobs fail setup.

The fresh single-user application creates a disposable API key through its
installed management endpoint. Bootstrap is restricted to GitHub-hosted runners,
refuses an existing key, masks the new secret in Actions, verifies authentication
and writes an owner-only file. No local API key, repository secret or external
model account is required. The private runtime directory and storage database are
excluded from uploads. Generation logs and JUnit are scrubbed of the known API
key before parsing and hashing; a final scrub covers the report tree before
upload. A failed scrub prevents upload. Console masking alone is insufficient
for archive files. This setup is specifically for the disposable pinned
application, not a way to administer an existing production application.

The application container and Ubuntu runner have different UIDs. CI explicitly
enables read sharing for the fictional captured prompt files (0644) so the host
can evaluate them. The capture hook requires both the CI flag and the explicit
sharing flag; ordinary local captures stay owner-only (0600). API keys remain
0600 regardless of capture mode.

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
workflow branch as `main`. For a PR, enter its number in `pull-request` and leave
`revision` empty. The workflow resolves the open PR's current test merge commit,
which includes main and is the same candidate used by automatic PR checks.
For a standalone experiment, enter the full target SHA in `revision`. Leaving
both fields empty tests the dispatched main commit. Both fields cannot be set
together; closed PRs and PRs targeting another base branch are rejected.

From the repository root:

```bash
gh workflow run rag-live-quality.yml --ref main \
  -f revision="$(git rev-parse HEAD)" -f profile=curated -f models=primary
```

For the required pre-merge gate, replace `123` with the actual PR number:

```bash
gh workflow run rag-live-quality.yml --ref main \
  -f pull-request=123 -f profile=curated -f models=primary
```

PR changes, a new main commit or a merge conflict require resolving and testing
the new merge candidate. A successful standalone head-SHA experiment is not
automatically a pass for a different PR merge commit.

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
Statuses target the tested SHA, not the workflow's main SHA. The PR-number mode
targets its test merge SHA because GitHub may require results there when automatic
PR checks already exist on that candidate. Any subsequent code
commit requires new evidence. Concurrent runs for the same target are serialized.

Repository branch protection was configured and read back on 2026-10-10 with
these required contexts on `main`, all bound to the GitHub Actions app (15368):

- `Ruff and unit tests`;
- `Offline Page Object checks`;
- `RAG benchmark quality`.

The two automatic job checks use the GitHub Actions app as their expected source.
The AI context is a commit status written using the workflow's GitHub token.
The branch requires being current with main and disallows force pushes and
deletion. Administrator bypass remains available for deliberate repository
maintenance; a direct administrator push is not evidence that quality passed.
The repository setting is operational state, not implied by this YAML. Inspect
the branch protection API to confirm its current required contexts.

The semantic thresholds remain **experimental**. The gate demonstrates an
enforced engineering acceptance mechanism, not calibrated release suitability,
clinical accuracy or GxP approval. Disputed scores remain failed until reviewed;
neither workflow nor report builder changes thresholds to obtain a green run.

## Verification

Offline verification passed 599 unit cases with 81.21% combined statement/branch
coverage, above the 75% floor. New checks cover all four workload presets,
commit/source/baseline/plan/model provenance, preserved generation errors,
bootstrap refusal on developer machines, private key creation, private versus
explicitly shared capture permissions and checksum/size-verified blob downloads.
Ruff, isort, YAPF, actionlint 1.7.12 and Compose configuration validation passed.

The actual resolver shell was also executed against a stubbed GitHub CLI for
eight cases: default main, explicit SHA, open PR merge SHA, closed PR, another
base branch, absent merge candidate, conflicting selectors and invalid PR number.
Invalid inputs failed before publishing a status or invoking model setup.

Actual hosted execution and repository protection are verified separately after
publication; a local unit pass must not be reported as a passing live AI run.

### First hosted execution

[Run 38030127070](https://github.com/karmise/local-llm-lab/actions/runs/38030127070)
tested commit `42d92701bb0aaea37b08da46dea4bde7b80a7693`. The framework and offline
browser workflow passed. The live producer failed model preflight before any
answer/judge inference: the public Qwen3.5 tag had changed. The registry manifest
was `d8b0f5e9760cd1682034f292d7ef72ec46f432149be0df7574bf2d6e92e38c04`, while
the reviewed local model was `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`.

The downstream gate still ran, rejected missing complete benchmark evidence,
generated Allure HTML and retained both artifacts. The final commit status was
`RAG benchmark quality: failure`, written by `github-actions[bot]`. The disposable
services were removed. This demonstrates failure propagation and evidence
preservation; it is not a failed answer-quality experiment.

The follow-up installs the original manifests and their content-addressed blobs,
whose availability was checked in the official registry. It also handles the
container/host capture permissions explicitly. It does not change the reviewed
model weights, references or thresholds to make that first run pass.

### Full hosted execution and runtime correction

[Run 38030694328](https://github.com/karmise/local-llm-lab/actions/runs/38030694328)
tested `675a604d1811d03fbb8f44490f22f408658fea32` using the frozen manifests.
Installation and setup passed. Four declared rows produced execution errors,
not four valid low quality scores: Ollama 0.40.2 changed the legacy manifest
identity after inference, and the receipt-boundary generation exceeded 600
seconds on the CPU runner. The gate independently recomputed an error outcome,
generated Allure HTML and published a failing commit status. Cleanup completed.

The follow-up pins Ollama 0.35.0, matching the local runtime for the reviewed
weights, instead of adopting the new runtime's migrated model identity. Judge
controls also showed disagreements; their labels and thresholds remain unchanged.
A small hosted smoke run verifies the corrected execution chain without repeating
the entire expensive matrix. Smoke evidence cannot satisfy the curated merge gate.

A failed-request traceback included the disposable application's API key in the
archived log and JUnit file. Actions console masking did not redact those files.
The affected producer artifact was deleted and local diagnostics sanitized; the
application and its storage had already been destroyed. New redaction occurs
before evidence hashing and again before upload, with offline checks for repeated
keys, unchanged answer bytes, failing diagnostics and the parsing/hash order.

### Actual PR merge-gate negative check

[Temporary PR #1](https://github.com/karmise/local-llm-lab/pull/1) changed one
reviewed model digest to an intentionally incorrect checksum. The main working
tree and production model lock were unchanged. The actual PR-number workflow
[run 38032172017](https://github.com/karmise/local-llm-lab/actions/runs/38032172017)
resolved and tested merge candidate `2639d00b1277f453e7c8796164a997e858d6c619`,
rather than dispatch commit `838d18534d36842fdb280552b2c8446c024f3efa` or PR head
`fc3a84448147305d452f9bdd93270a62d5a1a6bd`.

Manifest validation failed before blob downloads; application setup and inference
were skipped, with **zero generation/judge calls**. The independent saved gate
rejected missing completed evidence, generated Allure HTML and retained failure
artifacts. `RAG benchmark quality: failure` was published on the merge candidate.
The non-draft PR's API state was `mergeable_state: blocked`. The PR was then closed
without merging and its temporary branch deleted. This proves the negative
integration path and commit targeting; it does not claim a positive quality pass.

## References

- [GitHub: required-check events and head versus test merge commit](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks).
- [GitHub: commit statuses reflected in pull requests](https://docs.github.com/en/rest/commits/statuses).
- [GitHub: protected branches and administrator bypass](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches).
- [Ollama: Docker and memory/concurrency configuration](https://github.com/ollama/ollama/blob/main/docs/faq.mdx).
- [AnythingLLM 1.16.2 API-key management implementation](https://github.com/Mintplex-Labs/anything-llm/blob/v1.16.2/server/endpoints/system.js).
