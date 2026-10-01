# Initial Environment Verification

Recorded on 2026-10-01 using the original localized fictional policy corpus.
The repository now contains an English policy; these historical checks do not
claim that the English corpus has been indexed or evaluated.

| Check | Historical result |
| --- | --- |
| AnythingLLM browser UI and `/api/ping` | PASS; workspace Company Policy Lab, online=true |
| Local model connectivity | PASS; Qwen2.5 on GPU with an 8192-token context |
| Document upload and embedding | PASS; one document indexed in the workspace |
| Policy question and sources | PASS; 23 working days of leave, 12 calendar days notice |
| Persistence | PASS; compose down without `-v`, then up; workspace, documents and history preserved |
| Environment helper | PASS; shell syntax, Compose configuration and status checked |

Additional answers contained KGS 2700 daily travel allowance, KGS 8400 hotel
limit with receipt, and up to six carried-over working days to be used by
March 31. For the absent gym policy, the model did not invent an amount.
The initial UI leave answer reported 4.8 seconds and 22.53 tokens/second;
this is one observation, not a performance benchmark.

## Retained metadata

- `versions.json` and `ollama-models.json`: installed versions and model digests.
- `container-status.json`: captured container status; local only, excluded from Git.
- `document-upload.json`: upload metadata; local only, excluded from Git.

Localized raw responses, workspace/history exports and screenshots have been
archived outside the project. They were not rewritten as English model outputs.
This summary records historical observations; English-corpus verification is
still required. Full model context capture is also required before RAGAS;
see [context capture notes](context-for-ragas.md).
