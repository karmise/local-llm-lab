# Step 4: Temporary Workspace Lifecycle

The framework can now provision a workspace for an individual test and remove
it during fixture teardown. This prepares isolated application data for future
document and RAG scenarios. The server, models and API key are still shared.

## What changed

- `AnythingLLMClient.create_workspace(name, configuration)` sends POST
  `/api/v1/workspace/new` and returns the raw response.
- `AnythingLLMClient.delete_workspace(slug)` sends DELETE
  `/api/v1/workspace/{slug}` and returns the raw response.
- `workspace_configuration` loads the English settings from
  `config/workspace.json`.
- `temporary_workspace` creates a uniquely named `automation-<uuid>` workspace
  for each test and yields the creation result.
- `test_temporary_workspace_lifecycle` reads that workspace through the API and
  validates its identity and every requested configuration field.

The transport accepts request payloads; application endpoints remain in the
API client. Resource setup and cleanup remain in fixtures, with scenario
expectations selected by tests and implemented in the shared assertions module.

## Setup, test and teardown

1. Generate a unique workspace name and send the configuration.
2. Validate the creation response and obtain the server-returned slug.
3. Yield the workspace to the test.
4. The test retrieves it and verifies its identity and settings.
5. In `finally`, delete the created workspace.
6. Retrieve the slug again and assert that `workspace` is an empty array.

The configured `Company Policy Lab` workspace is not used as the temporary
resource. Cleanup targets only the uniquely named workspace returned by creation.
A cleanup failure is reported as a pytest teardown error instead of being hidden.

The fixture owns deletion: tests using it should not delete the workspace
themselves. A separate delete-operation scenario can later use a fixture with
explicit ownership tracking. Function-scoped HTTP clients remain open until
workspace cleanup completes because the workspace fixture depends on them.

In AnythingLLM v1.16.2, creation and deletion succeed with HTTP 200. Looking up a
missing workspace returns HTTP 200 with `workspace=[]`, rather than HTTP 404.
The teardown verifies that contract, not just the delete response status.

## Verification

Verified on 2026-10-01: **6 passed in 0.09s**, including all previous checks.
Local report: `automation/reports/step-04.xml`, relative to the repository root.

An additional temporary probe deliberately failed after fixture setup. The
result contained one expected test failure and zero teardown errors; the
fixture confirmed the workspace was absent afterwards. The probe source was
removed. Its report is local only: `automation/reports/cleanup-probe.xml`.

Cleanup runs for ordinary assertion failures and exceptions after setup.
It cannot guarantee cleanup after a killed process, an unavailable server, or
a create request that times out before returning the resource identity.
Uniquely named leftovers can be investigated without mixing them with the
main workspace. Automatic deletion of arbitrary matching workspaces is not
implemented.

## Run and inspect

From `automation`:

```bash
source .venv/bin/activate
python -m pytest tests/test_workspace.py -v
python -m pytest tests/test_workspace.py::test_temporary_workspace_lifecycle -v --setup-show
```

Read the two client operations, then the `temporary_workspace` fixture, then
the lifecycle test. `--setup-show` displays pytest fixture setup and teardown
order.

This step does not call a generation model or upload documents. Next: upload
the English policy, attach/index it in the temporary workspace, and verify
readiness for retrieval before asking a policy question.

## Contract reference

[Installed-version workspace endpoint source](https://github.com/Mintplex-Labs/anything-llm/blob/v1.16.2/server/endpoints/api/workspace/index.js)

Interactive API documentation: http://localhost:3001/api/docs
