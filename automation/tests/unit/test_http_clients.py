"""HTTP clients: one shared transport and the exact AnythingLLM and Ollama requests each operation sends."""

from unittest.mock import Mock
from urllib.parse import quote

import pytest
import requests

from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.core import http_client
from llm_testkit.core.http_client import HttpClient
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit

KEY = {"Authorization": "Bearer test-key"}


@pytest.fixture
def session(monkeypatch):
    """The requests session behind every HttpClient."""
    session = Mock()
    monkeypatch.setattr(http_client.requests, "Session", Mock(return_value=session))
    return session


@title("A request joins the base URL and path and applies the default timeout and no redirects")
def test_transport_request(session):
    response = HttpClient("https://example.test/root/", 5).request("GET", "/api/ping")

    assert response is session.request.return_value
    session.request.assert_called_once_with(
            method="GET", url="https://example.test/root/api/ping", timeout=5, allow_redirects=False)


@pytest.mark.parametrize(("base", "path", "url"), [
        pytest.param("http://localhost", "chat", "http://localhost/chat", id="relative-path"),
        pytest.param("http://localhost//", "//chat", "http://localhost/chat", id="extra-slashes")])
@title("Exactly one slash separates the base URL from the path [{param_id}]")
def test_transport_joins_url(session, base, path, url):
    HttpClient(base, 5).request("POST", path)

    assert session.request.call_args.kwargs["url"] == url


@title("A caller may override the timeout and redirects, and pass request options through")
def test_transport_request_overrides(session):
    HttpClient("http://localhost", 5).request("POST", "chat", timeout=120, allow_redirects=True, json={"message": "hi"})

    session.request.assert_called_once_with(
            method="POST", url="http://localhost/chat", timeout=120, allow_redirects=True, json={"message": "hi"})


@title("The context manager closes the session")
def test_transport_closes_session(session):
    with HttpClient("http://localhost", 5) as http:
        session.close.assert_not_called()

    assert isinstance(http, HttpClient)
    session.close.assert_called_once_with()


@title("A failed generation is raised once, without retry, and the session is still closed")
def test_transport_does_not_retry(session):
    failure = requests.Timeout("Model response timed out")
    session.request.side_effect = failure

    with pytest.raises(requests.Timeout) as caught, HttpClient("http://localhost", 5) as http:
        http.request("POST", "/chat")

    assert caught.value is failure
    session.request.assert_called_once()
    session.close.assert_called_once_with()


@title("Clients sharing one transport do not share authentication")
def test_shared_transport_does_not_leak_authentication(session):
    http = HttpClient("http://localhost", 5)

    AnythingLLMClient(http, api_key="test-key").verify_authentication()
    AnythingLLMClient(http).verify_authentication()

    assert [c.kwargs["headers"] for c in session.request.call_args_list] == [KEY, {}]


@title("The health check is an unauthenticated ping")
def test_health(session):
    response = AnythingLLMClient(HttpClient("http://localhost", 5), api_key="test-key").health()

    assert response is session.request.return_value
    session.request.assert_called_once_with(
            method="GET", url="http://localhost/api/ping", timeout=5, allow_redirects=False)


SLUG = "policy/lab"
ENCODED = quote(SLUG, safe="")


@pytest.mark.parametrize(("operation", "method", "path", "options"), [
        pytest.param(lambda api: api.verify_authentication(), "GET", "auth", {}, id="verify-authentication"),
        pytest.param(lambda api: api.get_workspace(SLUG), "GET", f"workspace/{ENCODED}", {}, id="get-workspace"),
        pytest.param(
        lambda api: api.update_workspace(SLUG, {"chatMode": "chat"}), "POST", f"workspace/{ENCODED}/update",
        {"json": {
        "chatMode": "chat"}}, id="update-workspace"),
        pytest.param(
        lambda api: api.create_workspace("temporary", {"chatModel": "qwen"}), "POST", "workspace/new",
        {"json": {
        "chatModel": "qwen",
        "name": "temporary"}}, id="create-workspace"),
        pytest.param(
        lambda api: api.create_workspace("temporary"), "POST", "workspace/new", {"json": {
        "name": "temporary"}}, id="create-workspace-without-configuration"),
        pytest.param(
        lambda api: api.delete_workspace(SLUG), "DELETE", f"workspace/{ENCODED}", {}, id="delete-workspace"),
        pytest.param(
        lambda api: api.create_document_folder("folder"), "POST", "document/create-folder",
        {"json": {
        "name": "folder"}}, id="create-folder"),
        pytest.param(
        lambda api: api.delete_document_folder("folder"), "DELETE", "document/remove-folder", {
        "json": {
        "name": "folder"},
        "timeout": 180}, id="delete-folder"),
        pytest.param(
        lambda api: api.delete_document_folder("folder", timeout=7), "DELETE", "document/remove-folder", {
        "json": {
        "name": "folder"},
        "timeout": 7}, id="delete-folder-timeout"),
        pytest.param(
        lambda api: api.get_document_folder("a/b c"), "GET", "documents/folder/a%2Fb%20c", {}, id="get-folder"),
        pytest.param(
        lambda api: api.add_workspace_documents(SLUG, ["folder/policy.json"]), "POST",
        f"workspace/{ENCODED}/update-embeddings", {
        "json": {
        "adds": ["folder/policy.json"],
        "deletes": []},
        "timeout": 180}, id="index-documents"),
        pytest.param(
        lambda api: api.add_workspace_documents(SLUG, [], timeout=9), "POST", f"workspace/{ENCODED}/update-embeddings",
        {
        "json": {
        "adds": [],
        "deletes": []},
        "timeout": 9}, id="index-documents-timeout"),
        pytest.param(
        lambda api: api.search_workspace(SLUG, "leave?"), "POST", f"workspace/{ENCODED}/vector-search", {
        "json": {
        "query": "leave?",
        "topN": 4,
        "scoreThreshold": 0.25},
        "timeout": 180}, id="search"),
        pytest.param(
        lambda api: api.search_workspace(SLUG, "leave?", top_n=2, score_threshold=0.5, timeout=8), "POST",
        f"workspace/{ENCODED}/vector-search", {
        "json": {
        "query": "leave?",
        "topN": 2,
        "scoreThreshold": 0.5},
        "timeout": 8}, id="search-options"),
        pytest.param(
        lambda api: api.chat(SLUG, "hello"), "POST", f"workspace/{ENCODED}/chat", {
        "json": {
        "message": "hello",
        "mode": "query"},
        "timeout": 300}, id="chat"),
        pytest.param(
        lambda api: api.chat(SLUG, "hello", mode="chat", timeout=30), "POST", f"workspace/{ENCODED}/chat", {
        "json": {
        "message": "hello",
        "mode": "chat"},
        "timeout": 30}, id="chat-options")])
@title("Each AnythingLLM operation sends its authenticated developer API request [{param_id}]")
def test_anythingllm_operations(session, operation, method, path, options):
    response = operation(AnythingLLMClient(HttpClient("http://localhost", 5), api_key="test-key"))

    assert response is session.request.return_value
    session.request.assert_called_once_with(
            method=method, url=f"http://localhost/api/v1/{path}", headers=KEY, **({
            "timeout": 5} | options), allow_redirects=False)


@pytest.mark.parametrize("slug", ["a/b", "a b", "x?query#fragment", "non-ascii-\N{SNOWMAN}"])
@title("Workspace slugs are encoded as one path segment [{slug}]")
def test_workspace_slugs_are_one_path_segment(session, slug):
    AnythingLLMClient(HttpClient("http://localhost", 5)).get_workspace(slug)

    assert session.request.call_args.kwargs["url"] == f"http://localhost/api/v1/workspace/{quote(slug, safe='')}"


@title("Creating or updating a workspace does not mutate the given configuration")
def test_workspace_configuration_is_not_mutated(session):
    configuration = {"name": "template", "chatModel": "qwen"}
    api = AnythingLLMClient(HttpClient("http://localhost", 5))

    api.create_workspace("temporary", configuration)
    api.update_workspace("temporary", configuration)

    assert configuration == {"name": "template", "chatModel": "qwen"}
    assert session.request.call_args_list[0].kwargs["json"] == {"name": "temporary", "chatModel": "qwen"}


@pytest.fixture
def policy(tmp_path):
    path = tmp_path / "policy.txt"
    path.write_text("Fictional policy", encoding="utf-8")
    return path


def capture_upload(session, *, error=None):
    """Record the uploaded file object and its content when the request is sent."""
    uploads = []

    def upload(**kwargs):
        name, document, content_type = kwargs["files"]["file"]
        uploads.append({"name": name, "content": document.read(), "type": content_type, "document": document})
        if error:
            raise error
        return session.response

    session.response = requests.Response()
    session.request.side_effect = upload
    return uploads


@title("An upload sends the file as plain text under the folder, then closes it")
def test_upload_document(session, policy):
    uploads = capture_upload(session)

    response = AnythingLLMClient(HttpClient("http://localhost", 5),
            api_key="test-key").upload_document(policy, "my folder", filename="unique-policy.txt", timeout=42)

    upload, = uploads
    assert response is session.response
    assert (upload["name"], upload["content"],
            upload["type"]) == ("unique-policy.txt", b"Fictional policy", "text/plain")
    assert upload["document"].closed
    kwargs = session.request.call_args.kwargs
    assert (kwargs["method"], kwargs["url"], kwargs["headers"],
            kwargs["timeout"]) == ("POST", "http://localhost/api/v1/document/upload/my%20folder", KEY, 42)


@title("Without a filename the upload uses the file's own name and the default timeout")
def test_upload_document_defaults(session, policy):
    uploads = capture_upload(session)

    AnythingLLMClient(HttpClient("http://localhost", 5)).upload_document(policy, "folder")

    assert (uploads[0]["name"], session.request.call_args.kwargs["timeout"]) == ("policy.txt", 180)


@title("The uploaded file is closed even when the request fails")
def test_upload_closes_document_on_failure(session, policy):
    uploads = capture_upload(session, error=requests.Timeout("Upload timed out"))

    with pytest.raises(requests.Timeout, match="Upload timed out"):
        AnythingLLMClient(HttpClient("http://localhost", 5)).upload_document(policy, "folder")

    assert uploads[0]["document"].closed


@title("Listing Ollama models requests the tags endpoint")
def test_ollama_list_models(session):
    response = OllamaClient(HttpClient("http://localhost:11434", 5)).list_models()

    assert response is session.request.return_value
    session.request.assert_called_once_with(
            method="GET", url="http://localhost:11434/api/tags", timeout=5, allow_redirects=False)


@pytest.mark.parametrize(("timeout", "expected"),
        [pytest.param({}, 300, id="default"),
        pytest.param({"timeout": 30}, 30, id="given")])
@title("A structured chat asks once, without streaming or thinking, for output matching the schema [{param_id}]")
def test_ollama_structured_chat(session, timeout, expected):
    OllamaClient(HttpClient("http://localhost:11434", 5)).structured_chat(
            model="judge", prompt="Judge this", schema={"type": "object"}, options={"temperature": 0}, **timeout)

    session.request.assert_called_once_with(
            method="POST", url="http://localhost:11434/api/chat", timeout=expected, allow_redirects=False, json={
            "model": "judge",
            "stream": False,
            "think": False,
            "keep_alive": "1m",
            "messages": [{
            "role": "user",
            "content": "Judge this"}],
            "format": {
            "type": "object"},
            "options": {
            "temperature": 0}})
