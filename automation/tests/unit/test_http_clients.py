from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.core.http_client import HttpClient

pytestmark = pytest.mark.unit


@pytest.fixture
def session(monkeypatch: pytest.MonkeyPatch) -> Mock:
    session = Mock(spec=requests.Session)
    monkeypatch.setattr(requests, "Session", lambda: session)
    return session


def test_transport_preserves_response_timeouts_and_redirect_contract(session: Mock) -> None:
    with HttpClient("https://example.test/root/", 5) as http:
        response = http.request("GET", "/api/ping")
        assert response is session.request.return_value
        session.request.assert_called_once_with(
            method="GET",
            url="https://example.test/root/api/ping",
            timeout=5,
            allow_redirects=False,
        )
        http.request("POST", "chat", timeout=120, allow_redirects=True, json={"message": "hello"})
        assert session.request.call_args.kwargs["timeout"] == 120
        assert session.request.call_args.kwargs["allow_redirects"] is True
    session.close.assert_called_once_with()


def test_transport_closes_and_does_not_retry_failed_generation(session: Mock) -> None:
    failure = requests.Timeout("Model response timed out")
    session.request.side_effect = failure
    with pytest.raises(requests.Timeout) as caught, HttpClient("http://localhost", 5) as http:
        http.request("POST", "/chat")
    assert caught.value is failure
    session.request.assert_called_once()
    session.close.assert_called_once_with()


def test_shared_transport_does_not_leak_authentication_between_clients(session: Mock) -> None:
    with HttpClient("http://localhost", 5) as http:
        authenticated = AnythingLLMClient(http, api_key="test-key")
        anonymous = AnythingLLMClient(http)
        authenticated.verify_authentication()
        anonymous.verify_authentication()
    assert session.request.call_args_list[0].kwargs["headers"] == {
        "Authorization": "Bearer test-key"
    }
    assert session.request.call_args_list[1].kwargs["headers"] == {}


@pytest.mark.parametrize("slug", ["a/b", "a b", "x?query#fragment", "non-ascii-\N{SNOWMAN}"])
def test_workspace_slugs_are_encoded_as_one_path_segment(session: Mock, slug: str) -> None:
    from urllib.parse import quote

    with HttpClient("http://localhost", 5) as http:
        AnythingLLMClient(http).get_workspace(slug)
    assert (
        session.request.call_args.kwargs["url"]
        == f"http://localhost/api/v1/workspace/{quote(slug, safe='')}"
    )


def test_workspace_creation_does_not_mutate_configuration(session: Mock) -> None:
    configuration = {"name": "template", "chatModel": "test-model"}
    with HttpClient("http://localhost", 5) as http:
        AnythingLLMClient(http).create_workspace("temporary", configuration)
    assert configuration["name"] == "template"
    assert session.request.call_args.kwargs["json"] == {
        "name": "temporary",
        "chatModel": "test-model",
    }


def test_workspace_update_uses_authenticated_route_and_preserves_settings(session: Mock) -> None:
    configuration = {"chatMode": "chat", "openAiPrompt": "Reviewed prompt"}
    with HttpClient("http://localhost", 5) as http:
        response = AnythingLLMClient(http, api_key="test-key").update_workspace(
            "policy/lab", configuration
        )
    assert response is session.request.return_value
    session.request.assert_called_once_with(
        method="POST",
        url="http://localhost/api/v1/workspace/policy%2Flab/update",
        headers={"Authorization": "Bearer test-key"},
        json={"chatMode": "chat", "openAiPrompt": "Reviewed prompt"},
        timeout=5,
        allow_redirects=False,
    )
    assert configuration == {"chatMode": "chat", "openAiPrompt": "Reviewed prompt"}


@pytest.mark.parametrize("fail", [False, True])
def test_upload_closes_document_even_when_request_fails(
    session: Mock, tmp_path: Path, fail: bool
) -> None:
    path = tmp_path / "policy.txt"
    path.write_text("Fictional policy", encoding="utf-8")
    documents = []

    def upload(**kwargs):
        filename, document, content_type = kwargs["files"]["file"]
        documents.append(document)
        assert filename == "unique-policy.txt"
        assert document.read() == b"Fictional policy"
        assert content_type == "text/plain"
        assert kwargs["timeout"] == 42
        if fail:
            raise requests.Timeout("Upload timed out")
        return requests.Response()

    session.request.side_effect = upload
    with HttpClient("http://localhost", 5) as http:
        api = AnythingLLMClient(http)
        if fail:
            with pytest.raises(requests.Timeout):
                api.upload_document(path, "folder", filename="unique-policy.txt", timeout=42)
        else:
            api.upload_document(path, "folder", filename="unique-policy.txt", timeout=42)
    assert len(documents) == 1
    assert documents[0].closed
