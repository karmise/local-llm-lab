from contextlib import contextmanager
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
from requests import Response

from llm_testkit import assertions
from llm_testkit.config import Settings

pytestmark = pytest.mark.unit


def _response(payload: dict, status: int = 200) -> Response:
    response = Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()
    return response


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_existing_resource_fixtures_cleanup_after_failed_quality_check(
    tmp_path: Path, cleanup_failure: bool,
) -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "conftest.py"
    module = next(m for m in list(sys.modules.values()) if getattr(m, "__file__", None) == str(fixture_path))
    api = Mock()
    actions = []
    api.create_workspace.side_effect = lambda name, config: _response({"workspace": {"slug": name, "id": 1}})
    api.create_document_folder.return_value = _response({"success": True})
    api.upload_document.side_effect = lambda path, folder, **kwargs: _response({
        "success": True, "error": None,
        "documents": [{"title": kwargs["filename"], "location": f"{folder}/policy.json"}],
    })

    def delete_folder(folder: str, **kwargs) -> Response:
        actions.append("document cleanup")
        if cleanup_failure:
            raise RuntimeError("Document cleanup unavailable")
        return _response({"success": True})

    def delete_workspace(slug: str) -> Response:
        actions.append("workspace cleanup")
        return _response({})

    api.delete_document_folder.side_effect = delete_folder
    api.delete_workspace.side_effect = delete_workspace
    api.get_document_folder.side_effect = lambda folder: _response({"folder": folder, "documents": []}, 404)
    api.get_workspace.return_value = _response({"workspace": []})
    policy = tmp_path / "policy.txt"
    policy.write_text("Fictional policy")
    workspace_scope = contextmanager(module.temporary_workspace.__wrapped__)
    document_scope = contextmanager(module.uploaded_policy_document.__wrapped__)
    expected = "Document cleanup unavailable" if cleanup_failure else "Quality check failed"
    with pytest.raises(RuntimeError, match=expected):
        with workspace_scope(api, {}) as workspace:
            with document_scope(api, workspace, Settings(), policy):
                raise RuntimeError("Quality check failed")
    assertions.assert_field_equals({"actions": actions}, "actions", ["document cleanup", "workspace cleanup"])
    assertions.assert_field_equals({"workspace_checks": api.get_workspace.call_count}, "workspace_checks", 1)
    assertions.assert_field_equals({"document_checks": api.get_document_folder.call_count}, "document_checks", 0 if cleanup_failure else 1)
