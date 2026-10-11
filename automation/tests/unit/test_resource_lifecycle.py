"""Resource fixtures: every created workspace and document folder is cleaned up, whatever fails."""

import json

import pytest

from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit

# A conftest whose AnythingLLM double fails at the step named by LIFECYCLE_FAILURE and records cleanup calls.
CONFTEST = """
import json
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from requests import Response

from llm_testkit.config import Settings

pytest_plugins = ["test_support.fixtures.resources"]
FAILURE = os.environ["LIFECYCLE_FAILURE"]
calls = []


def response(payload, status=200):
    result = Response()
    result.status_code = status
    result._content = json.dumps(payload).encode()
    return result


@pytest.fixture
def settings():
    return Settings(document_timeout=42.0)


@pytest.fixture
def workspace_configuration():
    return {"chatModel": "qwen"}


@pytest.fixture
def policy_file(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text("Fictional policy")
    return path


def record(name, payload, status=200, error=None):
    def call(*args, **kwargs):
        calls.append([name, [str(a) for a in args], {k: str(v) for k, v in kwargs.items()}])
        if error:
            raise error
        return response(payload(*args, **kwargs) if callable(payload) else payload, status)
    return call


@pytest.fixture
def authenticated_anythingllm_api():
    api = Mock()
    api.create_workspace.side_effect = record("create_workspace", lambda name, config: {"workspace": {
        "slug": "" if FAILURE == "empty-slug" else name, "id": 1,
        "name": "wrong-name" if FAILURE == "workspace-validation" else name}},
        500 if FAILURE == "workspace-status" else 200)
    api.create_document_folder.side_effect = record(
        "create_document_folder", {"success": FAILURE != "folder-validation"},
        500 if FAILURE == "folder-status" else 200)
    api.upload_document.side_effect = record("upload_document", lambda path, folder, **kw: {
        "success": True, "error": None,
        "documents": [{"title": kw["filename"], "location": f"{folder}/policy.json"}]},
        500 if FAILURE == "upload" else 200)
    api.add_workspace_documents.side_effect = record(
        "add_workspace_documents", lambda slug, paths, **kw: {"workspace": {"slug": slug}},
        500 if FAILURE == "indexing" else 200)
    api.delete_document_folder.side_effect = record(
        "delete_document_folder", {"success": FAILURE != "folder-cleanup-rejected"},
        error=RuntimeError("Document cleanup unavailable") if FAILURE == "folder-cleanup" else None)
    api.get_document_folder.side_effect = record(
        "get_document_folder", lambda folder: {"folder": folder, "documents": []},
        200 if FAILURE == "folder-remains" else 404)
    api.delete_workspace.side_effect = record("delete_workspace", {}, 500 if FAILURE == "workspace-cleanup" else 200)
    api.get_workspace.side_effect = record(
        "get_workspace", {"workspace": [{"slug": "kept"}] if FAILURE == "workspace-remains" else []})
    return api


def pytest_sessionfinish():
    Path("calls.json").write_text(json.dumps(calls))
"""

TEST = """
import os


def test_scenario(indexed_workspace):
    assert os.environ["LIFECYCLE_FAILURE"] != "test", "Quality check failed"
"""

FOLDER_CLEANUP = ["delete_document_folder", "get_document_folder"]
WORKSPACE_CLEANUP = ["delete_workspace", "get_workspace"]
SETUP = ["create_workspace", "create_document_folder", "upload_document", "add_workspace_documents"]


@pytest.fixture
def lifecycle(framework_pytester, monkeypatch):
    """Run the scenario with the given failure and return its outcomes and the API calls it made."""
    framework_pytester.makeconftest(CONFTEST)
    framework_pytester.makepyfile(TEST)

    def run(failure):
        monkeypatch.setenv("LIFECYCLE_FAILURE", failure)
        result = framework_pytester.runpytest("-q", "-p", "no:cacheprovider")
        calls = json.loads((framework_pytester.path / "calls.json").read_text())
        return result, calls

    return run


def names(calls) -> list[str]:
    return [name for name, _, _ in calls]


@title("A passing scenario creates, indexes and then removes the folder before the workspace")
def test_resources_are_created_and_cleaned_up(lifecycle):
    result, calls = lifecycle("none")

    result.assert_outcomes(passed=1)
    assert names(calls) == SETUP + FOLDER_CLEANUP + WORKSPACE_CLEANUP


@title("Setup and cleanup use one generated workspace name for the workspace, folder and document")
def test_resources_share_generated_name(lifecycle):
    _, calls = lifecycle("none")

    by_name = {name: (args, kwargs) for name, args, kwargs in calls}
    workspace = by_name["create_workspace"][0][0]
    assert workspace.startswith("automation-") and len(workspace) == len("automation-") + 32
    assert by_name["create_workspace"][0][1] == str({"chatModel": "qwen"})
    assert by_name["create_document_folder"][0] == [workspace]
    assert by_name["upload_document"][0][1:] == [workspace]
    assert by_name["upload_document"][1] == {"filename": f"{workspace}-company-policy.txt", "timeout": "42.0"}
    assert by_name["add_workspace_documents"] == ([workspace, str([f"{workspace}/policy.json"])], {"timeout": "42.0"})
    assert by_name["delete_document_folder"] == ([workspace], {"timeout": "42.0"})
    assert [by_name[name][0]
            for name in ("get_document_folder", "delete_workspace", "get_workspace")] == [[workspace]] * 3


@pytest.mark.parametrize(("failure", "created", "cleanup", "message"), [
        pytest.param("workspace-status", ["create_workspace"], [], "Workspace setup", id="workspace-status"),
        pytest.param("empty-slug", ["create_workspace"], [], "nonempty slug for cleanup", id="empty-slug"),
        pytest.param(
        "workspace-validation", ["create_workspace"], WORKSPACE_CLEANUP, "wrong-name", id="workspace-validation"),
        pytest.param("folder-status", SETUP[:2], WORKSPACE_CLEANUP, "Test document folder setup", id="folder-status"),
        pytest.param(
        "folder-validation", SETUP[:2], FOLDER_CLEANUP + WORKSPACE_CLEANUP, "Test document folder setup",
        id="folder-validation"),
        pytest.param("upload", SETUP[:3], FOLDER_CLEANUP + WORKSPACE_CLEANUP, "500", id="upload"),
        pytest.param("indexing", SETUP, FOLDER_CLEANUP + WORKSPACE_CLEANUP, "500", id="indexing")])
@title("A setup failure is an error that still removes everything created before it [{param_id}]")
def test_setup_failure_cleans_up_created_resources(lifecycle, failure, created, cleanup, message):
    result, calls = lifecycle(failure)

    result.assert_outcomes(errors=1)
    assert names(calls) == created + cleanup
    result.stdout.fnmatch_lines([f"*{message}*"])


@title("A failing test still removes the folder and then the workspace")
def test_test_failure_cleans_up(lifecycle):
    result, calls = lifecycle("test")

    result.assert_outcomes(failed=1)
    assert names(calls) == SETUP + FOLDER_CLEANUP + WORKSPACE_CLEANUP
    result.stdout.fnmatch_lines(["*Quality check failed*"])


@pytest.mark.parametrize(("failure", "cleanup", "message"), [
        pytest.param(
        "folder-cleanup", ["delete_document_folder"] + WORKSPACE_CLEANUP, "Document cleanup unavailable",
        id="folder-delete-raises"),
        pytest.param(
        "folder-cleanup-rejected", ["delete_document_folder"] + WORKSPACE_CLEANUP, "Document cleanup for",
        id="folder-delete-rejected"),
        pytest.param(
        "folder-remains", FOLDER_CLEANUP + WORKSPACE_CLEANUP, "Document folder cleanup for automation-",
        id="folder-remains"),
        pytest.param(
        "workspace-cleanup", FOLDER_CLEANUP + ["delete_workspace"], "Cleanup for automation-",
        id="workspace-delete-fails"),
        pytest.param(
        "workspace-remains", FOLDER_CLEANUP + WORKSPACE_CLEANUP, "Field workspace: expected length 0, got 1",
        id="workspace-remains")])
@title("A cleanup failure is a teardown error; the workspace is still removed after a folder failure [{param_id}]")
def test_cleanup_failure_is_reported(lifecycle, failure, cleanup, message):
    result, calls = lifecycle(failure)

    result.assert_outcomes(passed=1, errors=1)
    assert names(calls) == SETUP + cleanup
    result.stdout.fnmatch_lines([f"*{message}*"])
