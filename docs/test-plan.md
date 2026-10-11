# Policy application test plan

Plan baseline: `policy-lab-plan-v1`, defined by
[`qualification-plan.json`](../automation/test_data/qualification-plan.json).
Approval status: draft, pending human review. Scope: an educational local RAG
application answering questions about a fictional company policy.

## Intended use and scope

AnythingLLM retrieves policy passages and generates answers through local Ollama
models. Users should receive the reviewed entitlements, supporting sources, and
an acknowledgement when information is absent. The framework validates the API,
document lifecycle, retrieval, generated answers, browser behavior and saved
evaluation evidence. It also checks the framework itself independently.

This plan does not qualify a clinical application, infrastructure deployment,
regulated manufacturing process or production service. It contains no patient
data. Policy correctness is anchored to the fictional source, rather than to
employment law. Model scores and curated bias/security checks have limited scope.

## Risks and acceptance

The machine-readable plan contains 13 requirements. Each has a risk, acceptance
criteria, an IQ/OQ/PQ phase, actual pytest selectors and any required matrix axes.
The mapping is validated against test definitions before execution. Changes to
policy/catalog/gate files require reviewing the baseline checksums in that plan.

The [dataset benchmark](step-33-dataset-benchmark.md) is a supplemental bounded
experiment, not an expansion of the declared requirement matrix. It evaluates
all four semantic metrics for selected positive golden cases, reports refusal
acceptance separately and preserves failed/missing cases in its planned denominator.
Its gate thresholds remain experimental and its human review remains pending.

| Risk | Check | Acceptance and limitation |
| --- | --- | --- |
| Unavailable or incompatible runtime | IQ: runtime, health, model inventory | Python 3.12, pinned base libraries, valid health response, both model names with digests; deployment qualification is outside this protocol |
| Unauthorized API access | OQ: authentication | Valid key accepted, absent/invalid keys rejected; not a complete security assessment |
| Leaked test state or unavailable retrieval | OQ: lifecycle and indexing | Owned workspaces/documents, verified cleanup, attached policy and searchable anchors; teardown errors remain deviations |
| Invented or incorrect entitlements | PQ: golden answers | 16 source-bound cases × two models; facts, prohibited claims, missing information and sources |
| Prompt changes degrade answers | PQ: prompt regression | The same 16 cases × two prompts × two models; matching configuration required for controlled comparisons |
| Prompt/document attacks override policy | PQ: adversarial | Six attacks × two models; actual captured exposure required for document attacks |
| Employee descriptors change uniform entitlement | PQ: paired bias checks | Three pairs × two variants × two models; both variants satisfy the same criteria; not population fairness validation |
| Browser behavior hides or loses evidence | PQ: UI | Four application scenarios: answer/source visibility, missing policy, source details and persisted history |
| Semantic quality degrades | PQ: saved quality gates | Paid-leave facts/sources plus four validated metrics; faithfulness and context precision/recall pass explicit experimental thresholds and factual correctness is recorded without one; not a dataset-wide accuracy estimate |
| Declared workloads degrade latency/correctness | PQ: performance | Both health and RAG batches meet their recorded latency/error gates; small bounded workloads, not capacity planning |

## Test strategy

Use cheap deterministic tests first: offline units validate client behavior,
fixtures, evidence binding, comparison logic and failure exits. Offline browser
checks protect Page Object waiting/selection. API tests check application
contracts and indexing separately from generation. UI tests reuse API setup so
browser failures remain easier to diagnose.

Non-deterministic answers are compared with reviewed facts/references and source
criteria, never with exact answer wording. Golden patterns are conservative smoke
checks. RAGAS adds faithfulness, factual correctness, context precision and recall
using captured model inputs. These dimensions answer different questions; a high
score in one dimension cannot replace a missing or failing check in another.

Compare prompt variants and employee-descriptor pairs under controlled settings.
Record model digests, dataset/prompt hashes and judge settings. History comparisons
flag changed conditions, duplicate observations and metric drops against an
explicit baseline. They do not establish statistical drift or identify its cause.
Do not retry a failed generation until it passes. Keep repetitions explicit and
review variability as evidence.

## Environment and data

Baseline application: AnythingLLM 1.16.2; revalidate selectors/capture parsing on
upgrade. Baseline runtime: Python 3.12 with pinned dependency locks. Local models:
`qwen3.5:4b` and `qwen2.5:7b`; names alone do not identify weights, so record digests.
Configuration comes from `config/workspace.json` and documented environment
variables. Credentials and generated reports remain ignored by Git.

The policy, golden cases, prompt variants, adversarial and bias catalogs, quality
profiles and thresholds are version-controlled. Policy/catalog hashes bind
expectations to the source. Fixtures own temporary workspaces and uploaded
documents; cleanup is registered as soon as ownership is established and verifies
absence. A creation timeout or missing identifier can prevent reliable cleanup.

## Execution, entry and exit criteria

Entry: reviewed scope and baselines, installed dependencies, reachable service,
declared models, valid developer key, and a generation/judge budget. Begin with
one case/model. Full matrices remain opt-in and require separate deliberate runs;
the qualification mapping does not schedule them automatically.

Normal CI runs Ruff, offline units with a 75% combined statement/branch coverage
floor, and offline Page Objects. Failed checks return nonzero. The separate saved
quality-gate workflow requires producer artifacts. Hosted CI does not run local
models; configuring required branch checks remains a repository administration
step.

Protocol exit: every declared cell within the selected phases passed, cleanup
succeeded, no unbound/skipped/missing/error observations, and deviations reviewed.
An execution pass leaves human review pending. A small passing subset cannot
close the full PQ matrix. Results from other runs cannot erase recorded failures.

## Evidence and change control

JUnit records requirement IDs, node identity, plan/code hashes and relevant
scenario metadata. Optional Allure labels support navigation. The evidence
builder copies selected JUnit, definitions and explicitly selected report files,
then writes traceability, deviations, a readable summary and checksum manifest.
Checksums detect corruption against that manifest; they are not authenticated
signatures or a tamper-proof audit trail.

Before changing a model, prompt, dataset or gate, document the intended change and
hold other conditions fixed where possible. Review acceptance criteria rather
than relaxing them to accommodate a new output. Update baselines deliberately,
rerun affected checks and retain previous evidence separately. A reviewer should
confirm intended use, expected results, budget, deviations and the meaning of any
threshold. This project does not invent approval names, signatures or dates.

Follow the [educational execution protocol](step-31-qualification-evidence.md).
