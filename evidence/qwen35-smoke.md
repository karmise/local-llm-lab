# Second Generation Model Verification

Recorded on 2026-10-01 using the original localized corpus.

- Added `qwen3.5:4b`, Q4_K_M; retained `qwen2.5:7b`.
- Digest: `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`.
- Selected Qwen3.5 globally and for Company Policy Lab.
- Observed 100% GPU use, 8192-token context, approximately 3.2 GB loaded size.
- Preserved BGE-M3 and the existing document index.
- Verified helper switching to Qwen2.5 and back to Qwen3.5.

| Historical RAG check | Result |
| --- | --- |
| Paid leave and request notice | PASS; 23 working days, 12 calendar days, two sources |
| Missing gym reimbursement policy | PASS; reported insufficient information, no invented amount, two sources |
| API/model errors in these answers | None observed |

Application generation metrics were approximately 20.2 seconds for leave and
48.5 seconds for the gym question. Both responses included thinking. These are
single generation measurements, not end-to-end benchmarks. The initial Qwen2.5
UI observation was 4.8 seconds for the leave question; thinking mode must be
considered in future comparisons.

Model metadata remains in `versions-qwen35.json` and `ollama-models-qwen35.json`.
Localized raw responses and the screenshot were archived outside the project.
The English corpus has not been evaluated by these historical runs. These checks
establish connectivity, not general accuracy or test flakiness.
