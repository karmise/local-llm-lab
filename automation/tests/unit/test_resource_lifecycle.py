"""Exercise resource lifecycles through pytest, including setup and teardown failures."""

import json

import pytest

from test_support.builders.resource_lifecycle import (
    prepare_resources_are_cleaned_up_after_each_failure_case,
)
from test_support.data.resource_lifecycle import (
    RESOURCES_ARE_CLEANED_UP_AFTER_EACH_FAILURE_FAILURE_FAILED_ERRORS_CLEANUP_CASES,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("failure", "failed", "errors", "cleanup"),
    RESOURCES_ARE_CLEANED_UP_AFTER_EACH_FAILURE_FAILURE_FAILED_ERRORS_CLEANUP_CASES,
)
def test_resources_are_cleaned_up_after_each_failure(
    framework_pytester: pytest.Pytester,
    failure: str,
    failed: int,
    errors: int,
    cleanup: list[str],
) -> None:
    runner = framework_pytester
    runner.makeconftest(
        """
        import json
        from pathlib import Path
        from unittest.mock import Mock
        import pytest
        from requests import Response
        from llm_testkit.config import Settings

        pytest_plugins = ["test_support.fixtures.resources"]
        FAILURE = FAILURE_VALUE
        actions = []

        def response(payload, status=200):
            result = Response()
            result.status_code = status
            result._content = json.dumps(payload).encode()
            return result

        @pytest.fixture
        def settings():
            return Settings()

        @pytest.fixture
        def workspace_configuration():
            return {}

        @pytest.fixture
        def policy_file(tmp_path):
            path = tmp_path / "policy.txt"
            path.write_text("Fictional policy")
            return path

        @pytest.fixture
        def authenticated_anythingllm_api():
            api = Mock()
            api.create_workspace.side_effect = lambda name, config: response({"workspace": {
                "slug": name, "id": 1,
                "name": "wrong-name" if FAILURE == "workspace-validation" else name,
            }})
            api.create_document_folder.return_value = response({"success": FAILURE != "folder-validation"})
            api.upload_document.side_effect = lambda path, folder, **kw: response({
                "success": True, "error": None,
                "documents": [{"title": kw["filename"], "location": f"{folder}/policy.json"}],
            }, 500 if FAILURE == "upload" else 200)
            api.add_workspace_documents.side_effect = lambda slug, paths, **kw: response(
                {"workspace": {"slug": slug}}, 500 if FAILURE == "indexing" else 200,
            )

            def delete_folder(folder, **kwargs):
                actions.append("folder")
                if FAILURE == "cleanup":
                    raise RuntimeError("Document cleanup unavailable")
                return response({"success": True})

            def delete_workspace(slug):
                actions.append("workspace")
                return response({})

            api.delete_document_folder.side_effect = delete_folder
            api.delete_workspace.side_effect = delete_workspace
            api.get_document_folder.side_effect = lambda folder: response({"folder": folder, "documents": []}, 404)
            api.get_workspace.return_value = response({"workspace": []})
            return api

        def pytest_sessionfinish():
            Path("cleanup.json").write_text(json.dumps(actions))
    """.replace("FAILURE_VALUE", repr(failure))
    )
    runner.makepyfile(
        """
        def test_scenario(indexed_workspace):
            FAILURE = FAILURE_VALUE
            assert FAILURE not in {"test", "cleanup"}, "Quality check failed"
    """.replace("FAILURE_VALUE", repr(failure))
    )

    result = runner.runpytest_subprocess("-q", "--tb=short")

    result.assert_outcomes(passed=int(failure == "none"), failed=failed, errors=errors)
    assert json.loads((runner.path / "cleanup.json").read_text()) == cleanup
    prepare_resources_are_cleaned_up_after_each_failure_case(failure, result)
