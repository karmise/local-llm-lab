# Step 3: Developer API Authentication and Workspace Access

The framework now calls the documented AnythingLLM developer API. This step
adds authentication checks and reads the existing configured workspace.
It does not create or delete workspaces or call the generation model.

## API contract

Verified against installed AnythingLLM v1.16.2:

| Operation | Endpoint | Expected result |
| --- | --- | --- |
| Check application availability | GET `/api/ping` | HTTP 200, `online=true`; no key required |
| Verify a valid key | GET `/api/v1/auth` | HTTP 200, `authenticated=true` |
| Verify without a key | GET `/api/v1/auth` | HTTP 403, `No valid api key found.` |
| Verify an invalid key | GET `/api/v1/auth` | HTTP 403, same error |
| Read a workspace | GET `/api/v1/workspace/{slug}` | HTTP 200, `workspace` array containing the matching workspace |

The developer API uses `Authorization: Bearer <key>`. In this version, rejected
keys return 403. The `workspace` response field is an array, including when
looking up one workspace. The test therefore checks that exactly one entry
matches the configured slug and has a positive integer ID.

## Configuration and key storage

A dedicated local key named `automation-framework` was created for the current
environment and saved in `.runtime/anythingllm-api-key` at the repository root.
The file contains only the key, has owner-only read/write permissions, and is
excluded from Git. Tests do not create or rotate keys automatically.

Configuration precedence:

1. Non-empty `ANYTHINGLLM_API_KEY` environment variable.
2. File supplied through `ANYTHINGLLM_API_KEY_FILE`.
3. Local default `.runtime/anythingllm-api-key`, supplied by the settings fixture.

The optional key is excluded from the `Settings` representation. Authenticated
fixtures fail with a configuration message if no key is available. Health and
negative authentication tests remain usable without a valid key. An explicitly
configured missing key file is an error instead of silently falling back.

`ANYTHINGLLM_WORKSPACE_SLUG` defaults to `company-policy-lab`. This is an
environment check against an existing workspace, not an isolated workspace
lifecycle test. The workspace must exist before this test is run.

On another computer, generate a key in the application's API key settings and
store it in a private file. Use `ANYTHINGLLM_API_KEY_FILE` to select that file.
Never commit the real key or insert it into examples, screenshots or reports.
API keys grant access to the application; they are not read-only test credentials.

## Layer responsibilities

- `Settings`: loads the optional key and workspace slug.
- `HttpClient`: remains generic and knows nothing about API keys.
- `AnythingLLMClient`: adds the Bearer header to developer API requests only;
  exposes `verify_authentication()` and `get_workspace(slug)`.
- Fixtures: provide either an unauthenticated or authenticated application client.
- Assertions: reusable API response and field checks.
- Tests: invoke named response checks, including expected rejection responses.

`anythingllm_api` intentionally remains unauthenticated. Positive developer API
tests request `authenticated_anythingllm_api`. The invalid-key test constructs
a client with the public dummy value `invalid-test-key`; it does not modify
shared session headers. Slugs are encoded as URL path segments in the client.

## Run locally

From `automation` with the application running:

```bash
source .venv/bin/activate
python -m pytest -v
python -m pytest -m api -v
python -m pytest tests/test_authentication.py -v
```

With an explicitly selected key file:

```bash
ANYTHINGLLM_API_KEY_FILE=/absolute/path/to/private-key-file python -m pytest -m api -v
```

The current local default works in PyCharm with the existing interpreter;
the settings fixture resolves its path independently of the working directory.
Use run configuration environment variables to select another key or workspace.

Verified on 2026-10-01: **5 passed in 0.04s** against the local application.
Local JUnit report: `automation/reports/step-03.xml`, relative to the repository
root. No generation request was made and the existing workspace was not changed.

## Next step

Create an isolated test workspace and clean it up with a yield fixture. Then add
document upload, indexing and an English policy question. The English repository
templates have not automatically migrated the existing application corpus.

## References

- [AnythingLLM API access](https://docs.anythingllm.com/features/api)
- Instance endpoint documentation: http://localhost:3001/api/docs
- [Authentication middleware, v1.16.2](https://github.com/Mintplex-Labs/anything-llm/blob/v1.16.2/server/utils/middleware/validApiKey.js)
- [Workspace endpoints, v1.16.2](https://github.com/Mintplex-Labs/anything-llm/blob/v1.16.2/server/endpoints/api/workspace/index.js)
