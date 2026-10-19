# Design decisions

The decisions that shape this framework, why they were made and where they live
in the code. Read them before changing a threshold, adding a retry or reshaping
a test layer: most of them were added after a concrete failure.

## Scope

### The corpus is deliberately small; the framework is the product

The policy (`data/company-policy.txt`) is a few fictional rules: leave days,
deadlines, travel and hotel limits, one approver and an explicit absence of gym
benefits. It is small and fixed so that every expected answer can be checked by
hand and every live run fits a local or CPU budget. What is being built and
tested is the evaluation framework around it: capture, judges, controls, gates,
provenance and reporting.

The unit test count reflects the framework's code, not the corpus. A larger
corpus adds dataset rows and live runs, not unit tests.

**Known limit.** Each golden case is currently its own pytest test. That suits
16 cases; with thousands of scenarios, cases should become dataset rows evaluated
by a harness that reports aggregate metrics and thresholds per stratum (fact
lookup, boundary, refusal, adversarial), with a small risk-based subset on PRs
and the full set nightly or before release.

### The numbers are illustrative, not a release claim

Thresholds are labelled `experimental` in `data/quality-gates.json`. A green
gate shows that an enforced acceptance mechanism works on a scoped curated slice.
It is not a statistical accuracy estimate, a fairness validation or GxP approval,
and the docs avoid claiming otherwise.

## Test layers

### The smallest layer that can answer the question

Unit (mocked HTTP), offline browser (local HTML), API, RAG, golden, UI and
quality layers answer different questions and have different costs. A transport
or locator defect is reproduced offline before spending model calls on it.
Expensive suites are opt-in (`--run-*` flags in `plugins/options.py`).
Bare `python -m pytest` includes live API and RAG tests; `tests/unit` is the
offline command.

### Unit tests are plain pytest, verified by mutation testing

Unit tests use plain `assert` and `pytest.raises(..., match=...)`, not the
application `assertions` package, so a broken acceptance check cannot also
verify itself. Each rejection test names its rule and has a paired acceptance
test. Mutation testing (mutmut) showed which tests actually detect a broken
rule, and raised scores per domain from 3.8–81.9% to 85.6–100%. Data shared by
several modules comes from `builders`; anything used by one module
stays in it.

### One top-level folder per layer; tests hold only tests

`automation/` is laid out like a test automation repository rather than a
library: `api/`, `pages/`, `fixtures/`, `models/`, `data/`, `helpers/`,
`assertions/`, then the LLM-specific `evaluation/`, `reporting/` and
`qualification/`. A reader sees the layers at once instead of looking under
`src/`. `tests/` contains only test modules: a request shape, a setup step or a
lookup that a test needs lives in `api/` or `fixtures/`. Typed test cases are
called `models/`, not `datasets/`, because a top-level `datasets` package would
shadow the HuggingFace library that RAGAS imports.

### Layers, each built on a base layer

Tests call fixtures, resource APIs and Page Objects; those call a base layer;
the base layer calls the transport. `BaseClient` (`api/base_client.py`) owns
the base path, the API key header and one-segment path encoding, so
`WorkspacesApi`, `DocumentsApi` and `OllamaClient` only name operations.
`BasePage` (`pages/base_page.py`) owns the application URL, navigation and the
answer timeout, so `WorkspacePage` only holds locators and actions. A rule such
as "encode a slug as one path segment" therefore exists once. There is no base
test class: pytest fixtures already provide shared setup, and inheritance
between tests would hide it.

### Clients return raw responses; acceptance criteria live in one place

`api/` builds requests and returns responses unchanged; `assertions/` holds
the rules (`fields`, `api`, `answers`, `quality`, `ui`). API, UI and quality
scenarios share the same profiles in `data` and the same checks, so a rule
changes in one place.

### Resources clean up after themselves, and only their own

A resource fixture registers cleanup as soon as creation returns a usable
identifier, before validating anything else, and verifies absence afterwards
(`fixtures/resources.py`). If creation returns no usable identifier,
the framework does not guess and never deletes resources it cannot prove it owns.

## Answer quality

### Facts, not wording

Answers are checked for required facts, forbidden content and sources, not
compared with a fixed string; a model may paraphrase correctly. These regex
checks are smoke contracts: a negated sentence can still contain the right
number, which is why semantic metrics exist alongside them.

### A reference states only what the source states

The policy used to say that hotel accommodation is reimbursed "with a receipt",
while the golden reference said "only with a receipt". That gap was the
reviewers' reading, not the text, and both models tripped over it: the generator
reasoned for minutes about whether the information was sufficient, and the judge
rejected "you cannot claim without a receipt" as unsupported. The policy now
states the condition (version SM-2026-02). A golden reference must follow from
the source text alone; when it depends on an interpretation, fix the source or
the reference, not the check.

### Evaluate the context the model actually received

Cited sources are not the full prompt. The capture overlay (`compose.capture.yaml`,
`observation/`) records the real Ollama request after AnythingLLM's message
compression, and semantic metrics run on that context. A document-injection
attack counts only if capture shows the attack text reached the model.

### A judge is trusted only after it agrees with labelled controls

An LLM judge can be wrong. Every benchmark first runs hand-labelled controls:
supported paraphrases, wrong numbers, invented amounts, day-type substitutions.
Each labelled claim must map to exactly one extracted statement with the expected
verdict (`evaluation/control_match.py`); an equal average score is not enough.
Statements are compared after spelled-out integers become digits
(`helpers/number_words.py`), because the judge may write "five thousand" for 5000.
A control mismatch fails the whole run.

### Faithfulness is checked sentence by sentence by a dedicated model

RAGAS faithfulness asks the judge to rewrite the answer into statements and then
to verify them. Our 4B judge rewrote "the manager approves the request" as "the
manager is responsible for approving it" and rejected its own paraphrase, the
same way on every rerun of that answer. Re-judging every saved answer showed
three such false rejections among 17 faithful answers, and verdicts on the same
answer differed between a CI runner and a Mac. Faithfulness now splits the answer
into sentences in code (`helpers/claims.py`) and asks Bespoke-MiniCheck 7B, a
local model trained only to say whether a document supports a claim, about each
sentence (`evaluation/claim_checker.py`). On the same answers it accepted 17 of
17 faithful answers, caught all 8 deliberately wrong ones and named the wrong
sentence each time; Granite Guardian 2B accepted 11 of 17. The set is small and
labelled by one engineer, so the controls still run before every benchmark. The
other metrics keep the RAGAS judge. MiniCheck's licence is non-commercial, which
fits this educational lab but not a commercial product.

### Factual correctness is measured but not gated

On answers with one or two claims, a single error by the 4B judge moves
correctness F1 by 0.33 or more, and F1 also penalised true facts that the
reference omitted. Its minimum is therefore `null`: the score is recorded as
`measured (no threshold)` and reviewed when below 1, but cannot fail a run alone.
Faithfulness and context precision/recall keep thresholds. A gate file must
still contain at least one threshold (`reporting/gates.py`).

### Refusals are not scored as perfect

For missing-information cases, semantic metrics are N/A: neither zero, one nor a
judge error. Refusals are accepted by their own rules: the answer acknowledges
the gap, names the topic and proposes no amount.

## Reproducibility

### No retries; a repetition is an experiment

Neither HTTP, the judge nor the benchmark retries a failed call or answer.
Retrying until green would hide exactly the variability the suite measures.
`--rag-repeat N` instead runs independent repetitions with fresh resources, and
the stability report compares them. Timeouts are failures.

### Determinism where the model allows it

The benchmark workspace generates at temperature 0 (`config/workspace.json`); at
0.1 a correct but hedged answer once failed faithfulness. The judge uses
temperature 0, thinking disabled and a fixed seed. The manual conversation
workspace keeps 0.1. None of this makes outputs fully deterministic, which is why
answers are judged by facts and metrics, not exact text.

### The model gets the same input in every run

Temperature 0 helps only if the input is the same. Two test artefacts used to
change it in every run: the unique `automation-<id>` file name, which the model
saw as the source document and repeated in its answers, and the capture marker
appended to the workspace prompt. A gate once failed for that reason alone: the
answers were correct, but their wording shifted with the name and the judge
scored the new wording differently. The policy is now uploaded under a unique
file name, so concurrent uploads stay apart, with the document title
`company-policy.txt` (`fixtures/resources.py`). The capture hook removes the
marker before the call and records the request exactly as forwarded
(`observation/ollama-preload.cjs`). What ties the context to the test is the
test's own workspace, which holds only that document. One difference remains:
AnythingLLM stamps the upload time into each chunk header, and its file-upload
API cannot set it.

### Evidence is bound by checksums

The golden dataset, catalogs, gates and qualification plan are versioned and
cross-referenced by SHA256 (`golden_dataset_sha256`, `data_sha256`). Judge
reports are bound to the sample they scored, and history records carry model
digest, prompt, policy and judge checksums. Changing reference data therefore
forces a version bump and visible re-review. Checksums detect mismatches; they do
not protect against someone rewriting both data and checksum.

### Validation raises, it does not assert

Framework validation raises `ValueError` instead of using `assert`, because
`python -O` strips assert statements (`helpers/scores.py`, `evaluation/control_match.py`).
Malformed, incomplete or incomparable evidence is an error, never a pass.

## CI and merging

### Fresh evidence for every commit, revalidated independently

`Fresh RAG quality` generates new answers in disposable services for the exact
commit under review; earlier benchmarks are never reused. A second job downloads
that evidence and revalidates it with zero model calls; the commit status
`RAG benchmark quality` is published only if both jobs succeed. For a PR it marks
the merge commit, and any new commit needs a new run. A `smoke` run publishes a
different status, so it cannot satisfy the merge gate.

### The answer budget fits a long answer on a slow runner

qwen3.5 reasons before it answers, and AnythingLLM 1.16.2 passes Ollama neither
`think` nor an output limit, so answer length varies by case and run. One
boundary case has taken about 4,300 tokens, and hosted CPU runners generate 6 to
15 tokens per second. By default AnythingLLM also stops waiting for Ollama after
300 s, a limit of Node's `fetch`; the longest answer seen before, 295 s, was just
under it. The CI services therefore set `OLLAMA_RESPONSE_TIMEOUT` to 14 minutes
(`compose.ci.yaml`), and the client waits 15 (`ANYTHINGLLM_LLM_TIMEOUT` in the
live workflow), so the application reports its own error first. The budget only
separates a slow answer from a stuck one: answer time is reported, not gated,
and an answer that needs longer still fails.

### Automatic CI stays offline

Every push runs lint, formatting, unit tests with a coverage floor and offline
browser checks. Model runs are manual because a curated run takes about two
hours on a CPU runner.

### Pinned profiles, nothing implicit

Dependencies are pinned in separate lock files (base, dev, UI, reporting,
evaluation), so a unit task does not install a browser or judge stack. Tools
used only for one-off analysis, such as mutmut, stay out of the lock files.
Selectors and capture parsing target AnythingLLM 1.16.2 and must be revalidated
on upgrade.
