# Step 41: developer API contract and negative tests

`automation/tests/test_api_contract.py` checks how the AnythingLLM developer API behaves
outside the happy path. None of the requests reaches a model: authentication is checked
before any route runs, and unknown workspaces are rejected before generation, so the
28 tests finish in well under a second against a running application.

Run them with the other API checks:

```bash
python -m pytest -m 'smoke or api'
```

## What is covered

| Area | Contract |
| --- | --- |
| Authentication | All eight protected routes, including chat and vector search, answer 403 with exactly `{"error": "No valid api key found."}` for a missing or an invalid key, and an unauthenticated create leaves no workspace. |
| Unknown workspace | Lookup returns an empty list. Chat is aborted with 400, `type: abort`, no text, no sources and an error naming the workspace. Vector search is rejected with a message naming it. Delete, settings update and indexing are rejected with 400 and create nothing. |
| Malformed input | Invalid JSON for a new workspace is rejected with 400 and creates nothing. An upload without a file is rejected and stores no document. |
| Cleanup | An unknown document folder is reported absent (404, no documents), and removing it succeeds, so resource cleanup is idempotent. |

## Observed application behaviour (AnythingLLM 1.16.2)

- An upload request without a file answers **500 Internal Server Error** instead of a 4xx
  client error. The request is still rejected, so the test requires only a status of 400 or
  above and no stored document; an upgrade that returns 4xx keeps passing.
- An unknown `/api/v1/...` route answers **200 with the web application's HTML page**. A client
  must not treat HTTP 200 as success; the framework's `assert_json_object` rejects such a body,
  and a test pins that.
- Delete, update and indexing of an unknown workspace answer plain-text `Bad Request`, while
  chat and vector search answer JSON. Tests check the status for the plain-text routes and the
  JSON fields for the others.

## In CI

The `Fresh RAG quality` workflow runs the health, authentication and contract tests right
after it bootstraps the disposable application and before any model call, so a broken API
contract fails the run in seconds instead of after the benchmark. Their JUnit report is
written to `reports/ci/api/contract.xml`, redacted with the rest of the reports and kept in
the `fresh-rag-evidence` artifact.
