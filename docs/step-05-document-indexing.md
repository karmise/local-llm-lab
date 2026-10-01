# Step 5: Document Upload, Indexing and Retrieval

The framework now uploads the English policy into a unique document folder,
indexes it in a temporary workspace, and verifies retrieval of its facts.
This checks the document pipeline before adding model answer evaluations.

## Scenario

`test_policy_document_is_indexed_and_searchable` checks:

1. The uploaded document is attached to the temporary workspace by its returned
   document location.
2. Vector search returns passages from the uniquely named uploaded file.
3. Those passages contain `23 working days` and `12 calendar days`.

The test invokes the shared assertions module. It contains no inline assertions.
An HTTP 200 upload/indexing response is not sufficient: empty search results,
passages from another document, and a missing expected fact all fail validation.
This verifies document retrieval, not the generation model's answer quality.

## Layers and fixtures

The API client adds operations for folder creation/removal/listing, multipart
upload, embedding updates and vector search. Upload closes the local file handle
after sending the request. On this step, the uploaded fixture is a plain-text
policy document.

Fixture dependencies:

```text
authenticated_anythingllm_api
└── temporary_workspace
    └── uploaded_policy_document
        └── indexed_workspace
```

`uploaded_policy_document` creates a folder named after the workspace's unique
slug. The upload filename is unique too, avoiding filename collisions in the
collector when multiple scenarios run. Upload and indexing are deliberately
separate API requests so their outcomes can be distinguished.

`indexed_workspace` adds the returned document location to the workspace.
The installed API awaits document embedding before returning. The test then
checks the workspace association and executes vector search. There are no fixed
sleeps or automatic retries masking failed indexing.

## API contract

Verified against AnythingLLM v1.16.2:

| Operation | Endpoint |
| --- | --- |
| Create document folder | POST `/api/v1/document/create-folder` |
| Upload and process file | POST `/api/v1/document/upload/{folder}` |
| Add document to workspace | POST `/api/v1/workspace/{slug}/update-embeddings` |
| Search indexed passages | POST `/api/v1/workspace/{slug}/vector-search` |
| Remove folder and documents | DELETE `/api/v1/document/remove-folder` |
| Verify folder absence | GET `/api/v1/documents/folder/{folder}` |

Upload returns `success=true`, `error=null` and a `documents` array containing
processed document metadata. Indexing takes an `adds` list of locations returned
by upload. Vector search returns a `results` list with text and source metadata.
The current test uses topN 4 and score threshold 0.25.

## Cleanup

The document fixture enters `try/finally` before upload. After the test, or after
an upload/indexing/assertion failure, it removes the owned folder through the API.
In this version, folder removal purges its processed documents, associated
vector-cache files and workspace document associations. The fixture verifies
HTTP 404 and an empty document list when looking up the removed folder.

The underlying workspace fixture then removes its own workspace and verifies
its absence. The main Company Policy Lab and its original corpus are untouched.
All deletion targets are unique resources owned by the current test.
Cleanup errors remain visible as pytest teardown errors. An interrupted process
or unavailable server can still leave resources requiring investigation.

## Configuration and execution

Document processing, indexing, vector search and document cleanup use
`ANYTHINGLLM_DOCUMENT_TIMEOUT`, defaulting to 180 seconds. This permits initial
embedding-model loading without increasing the five-second health/API timeout.
As with other Requests timeouts, it is not a total scenario deadline.

From `automation` with the application and Ollama running:

```bash
source .venv/bin/activate
python -m pytest tests/test_documents.py -v
python -m pytest -v
```

Verified on 2026-10-01: **14 passed in 0.50s** (7 application checks and 7 unit
checks). The first standalone document scenario passed in 2.32s. These are test
run durations, not model generation benchmarks. Local reports are stored under
`automation/reports/` and excluded from Git.

A temporary failure probe verified cleanup after successful indexing followed
by an intentional assertion failure. Both the document folder and workspace
were confirmed absent during teardown. The probe source was removed; its report
is `automation/reports/document-cleanup-probe.xml`.

## Next step

Use `indexed_workspace` for an English policy question and validate the final
model answer and supporting sources. Add a missing-information scenario next,
then run the same criteria on both generation models.

## Installed-version source references

- [Document endpoints](https://github.com/Mintplex-Labs/anything-llm/blob/v1.16.2/server/endpoints/api/document/index.js)
- [Workspace endpoints](https://github.com/Mintplex-Labs/anything-llm/blob/v1.16.2/server/endpoints/api/workspace/index.js)
- [Folder purge implementation](https://github.com/Mintplex-Labs/anything-llm/blob/v1.16.2/server/utils/files/purgeDocument.js)
