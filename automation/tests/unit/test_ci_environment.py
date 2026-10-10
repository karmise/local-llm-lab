"""CI environment: bootstrap only a disposable hosted application and verify reviewed model weights."""

import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest
import requests

from llm_testkit.ci import environment
from llm_testkit.reporting.steps import title
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit

SECRET = "disposable-unit-test-key"


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")


@pytest.mark.parametrize(
        "variables", [
        pytest.param({"RUNNER_ENVIRONMENT": "github-hosted"}, id="local"),
        pytest.param({
        "GITHUB_ACTIONS": "false",
        "RUNNER_ENVIRONMENT": "github-hosted"}, id="not-actions"),
        pytest.param({"GITHUB_ACTIONS": "true"}, id="unknown-runner"),
        pytest.param({
        "GITHUB_ACTIONS": "true",
        "RUNNER_ENVIRONMENT": "self-hosted"}, id="self-hosted")])
@title("Bootstrap is refused outside a disposable GitHub-hosted runner [{param_id}]")
def test_require_hosted_environment_rejects(monkeypatch, variables):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("RUNNER_ENVIRONMENT", raising=False)
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match="restricted to a disposable GitHub-hosted runner"):
        environment.require_hosted_environment()


@title("A GitHub-hosted runner is accepted")
def test_require_hosted_environment_accepts(hosted):
    assert environment.require_hosted_environment() is None


def http_client(monkeypatch, *responses):
    """Replace the HTTP client with a context-managed double answering requests in order."""
    http = MagicMock()
    http.__enter__.return_value = http
    http.request.side_effect = list(responses)
    factory = Mock(return_value=http)
    monkeypatch.setattr(environment, "HttpClient", factory)
    return factory, http


@pytest.fixture
def no_sleep(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(environment.time, "sleep", sleep)
    return sleep


@pytest.mark.parametrize(("url", "path"), [
        pytest.param("http://127.0.0.1:3001", "/api/ping", id="anythingllm"),
        pytest.param("http://127.0.0.1:11434", "/api/version", id="ollama")])
@title("Readiness is probed on the service's health endpoint [{param_id}]")
def test_wait_for_service_probes_health_endpoint(monkeypatch, no_sleep, url, path):
    factory, http = http_client(monkeypatch, Mock(status_code=200))

    environment.wait_for_service(url)

    factory.assert_called_once_with(url, 5)
    http.request.assert_called_once_with("GET", path)
    no_sleep.assert_not_called()


@title("Readiness is retried every two seconds after errors and non-200 replies")
def test_wait_for_service_retries(monkeypatch, no_sleep):
    _, http = http_client(monkeypatch, requests.ConnectionError(), Mock(status_code=503), Mock(status_code=200))

    environment.wait_for_service("http://127.0.0.1:3001", attempts=3)

    assert (http.request.call_count, no_sleep.call_args_list) == (3, [call(2), call(2)])


@title("A service that never becomes ready fails after the declared attempts")
def test_wait_for_service_gives_up(monkeypatch, no_sleep):
    _, http = http_client(monkeypatch, *[Mock(status_code=503)] * 3)

    with pytest.raises(ValueError, match=r"^Service did not become ready: http://127.0.0.1:3001$"):
        environment.wait_for_service("http://127.0.0.1:3001", attempts=3)
    assert http.request.call_count == 3


@title("Readiness waits ninety attempts by default")
def test_wait_for_service_default_attempts(monkeypatch, no_sleep):
    _, http = http_client(monkeypatch, *[Mock(status_code=503)] * 90)

    with pytest.raises(ValueError, match="did not become ready"):
        environment.wait_for_service("http://127.0.0.1:11434")
    assert http.request.call_count == 90


@title("An unexpected error while probing is not swallowed")
def test_wait_for_service_propagates_unexpected_errors(monkeypatch, no_sleep):
    http_client(monkeypatch, RuntimeError("bug"))

    with pytest.raises(RuntimeError, match="bug"):
        environment.wait_for_service("http://127.0.0.1:3001")


def key_reply(payload):
    reply = Mock()
    reply.json.return_value = payload
    return reply


@pytest.fixture
def bootstrap_service(tmp_path, monkeypatch, hosted):
    """A fresh application that creates a key and authenticates it."""
    factory, http = http_client(monkeypatch, key_reply({"apiKey": {"secret": SECRET}, "error": None}))
    wait = Mock()
    monkeypatch.setattr(environment, "wait_for_service", wait)
    verification = key_reply({"authenticated": True})
    client = Mock()
    client.return_value.verify_authentication.return_value = verification
    monkeypatch.setattr(environment, "AnythingLLMClient", client)
    return SimpleNamespace(
            root=tmp_path, factory=factory, http=http, wait=wait, client=client, verification=verification,
            key=tmp_path / ".runtime/anythingllm-api-key")


@title("Bootstrap creates a disposable key, masks it in the log and saves it privately")
def test_bootstrap_saves_key_privately(bootstrap_service, capsys):
    environment.bootstrap(bootstrap_service.root)

    assert bootstrap_service.key.read_text() == SECRET
    assert bootstrap_service.key.stat().st_mode & 0o777 == 0o600
    assert bootstrap_service.key.parent.stat().st_mode & 0o777 == 0o700
    assert capsys.readouterr().out == f"::add-mask::{SECRET}\n"
    bootstrap_service.wait.assert_called_once_with("http://127.0.0.1:3001")
    bootstrap_service.factory.assert_called_once_with("http://127.0.0.1:3001", 10)
    bootstrap_service.client.assert_called_once_with(bootstrap_service.http, api_key=SECRET)
    bootstrap_service.http.request.assert_called_once_with(
            "POST", "/api/system/generate-api-key", json={"name": "disposable-ci"})


@title("Bootstrap reuses an existing private runtime directory that holds no key")
def test_bootstrap_reuses_runtime_directory(bootstrap_service):
    (bootstrap_service.root / ".runtime").mkdir(mode=0o700)

    environment.bootstrap(bootstrap_service.root)

    assert bootstrap_service.key.read_text() == SECRET


@title("Bootstrap refuses to replace an existing key")
def test_bootstrap_refuses_existing_key(bootstrap_service):
    environment.bootstrap(bootstrap_service.root)

    with pytest.raises(ValueError, match="Refusing to bootstrap an existing API key"):
        environment.bootstrap(bootstrap_service.root)
    assert bootstrap_service.key.read_text() == SECRET


@title("Bootstrap is refused outside a hosted runner before anything is created")
def test_bootstrap_requires_hosted_runner(bootstrap_service, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS")

    with pytest.raises(ValueError, match="disposable GitHub-hosted runner"):
        environment.bootstrap(bootstrap_service.root)
    assert not (bootstrap_service.root / ".runtime").exists()


@pytest.mark.parametrize(
        "payload", [
        pytest.param({
        "apiKey": {
        "secret": SECRET},
        "error": "quota"}, id="error-reported"),
        pytest.param({
        "apiKey": None,
        "error": None}, id="no-key"),
        pytest.param({
        "apiKey": {
        "secret": " "},
        "error": None}, id="blank-secret"),
        pytest.param({
        "apiKey": {
        "secret": 1},
        "error": None}, id="non-string-secret")])
@title("A failed or empty key creation is refused and nothing is saved [{param_id}]")
def test_bootstrap_rejects_failed_key_creation(tmp_path, monkeypatch, hosted, payload, capsys):
    http_client(monkeypatch, key_reply(payload))
    monkeypatch.setattr(environment, "wait_for_service", Mock())

    with pytest.raises(ValueError, match="Disposable API key creation failed"):
        environment.bootstrap(tmp_path)
    assert not (tmp_path / ".runtime/anythingllm-api-key").exists()
    assert capsys.readouterr().out == ""


@title("An HTTP error from key creation is raised before anything is saved")
def test_bootstrap_raises_http_error(tmp_path, monkeypatch, hosted):
    reply = key_reply({})
    reply.raise_for_status.side_effect = requests.HTTPError("500")
    http_client(monkeypatch, reply)
    monkeypatch.setattr(environment, "wait_for_service", Mock())

    with pytest.raises(requests.HTTPError):
        environment.bootstrap(tmp_path)
    assert not (tmp_path / ".runtime/anythingllm-api-key").exists()


@pytest.mark.parametrize("authenticated", [pytest.param(False, id="false"), pytest.param("true", id="string")])
@title("A key the application does not authenticate is refused but kept for diagnostic redaction [{param_id}]")
def test_bootstrap_rejects_unauthenticated_key(bootstrap_service, authenticated):
    bootstrap_service.verification.json.return_value = {"authenticated": authenticated}

    with pytest.raises(ValueError, match="Disposable API key was not authenticated"):
        environment.bootstrap(bootstrap_service.root)
    assert bootstrap_service.key.read_text() == SECRET


@title("A failed verification request keeps the private key for diagnostic redaction")
def test_failed_verification_preserves_key(bootstrap_service):
    bootstrap_service.verification.raise_for_status.side_effect = requests.HTTPError("401")

    with pytest.raises(requests.HTTPError):
        environment.bootstrap(bootstrap_service.root)
    assert bootstrap_service.key.read_text() == SECRET
    assert bootstrap_service.key.stat().st_mode & 0o777 == 0o600


LOCK = {"qwen3.5:4b": "qwen-weights", "bge-m3:567m": "embedding-weights"}


@pytest.fixture
def model_service(tmp_path, monkeypatch):
    """A local Ollama listing the reviewed weights and reporting its version."""
    lock = tmp_path / "models.json"
    lock.write_text(json.dumps(LOCK))
    catalog = key_reply({"models": [{"name": name, "digest": digest} for name, digest in LOCK.items()]})
    client = Mock()
    client.return_value.list_models.return_value = catalog
    monkeypatch.setattr(environment, "OllamaClient", client)
    version = key_reply({"version": "0.9.0"})
    factory, http = http_client(monkeypatch, version)
    return SimpleNamespace(
            lock=lock, catalog=catalog, version=version, client=client, factory=factory, http=http,
            output=tmp_path / "env.json")


@title("Verified generation and embedding weights and the Ollama version are recorded")
def test_verify_models_records_weights(model_service):
    environment.verify_models(model_service.lock, list(LOCK), model_service.output)

    assert json.loads(model_service.output.read_text()) == {"ollama_version": "0.9.0", "models": LOCK}
    model_service.factory.assert_called_once_with("http://127.0.0.1:11434", 10)
    model_service.client.assert_called_once_with(model_service.http)
    model_service.http.request.assert_called_once_with("GET", "/api/version")


@title("Only the selected models are recorded")
def test_verify_models_records_selection(model_service):
    environment.verify_models(model_service.lock, ["bge-m3:567m"], model_service.output)

    assert json.loads(model_service.output.read_text())["models"] == {"bge-m3:567m": "embedding-weights"}


@pytest.mark.parametrize(("selected", "listed"), [
        pytest.param(["qwen3.5:4b"], {"qwen3.5:4b": "changed"}, id="changed-digest"),
        pytest.param(["qwen3.5:4b"], {}, id="not-installed"),
        pytest.param(["llama3:8b"], {"llama3:8b": "weights"}, id="not-reviewed")])
@title("Changed, missing or unreviewed weights fail preflight and nothing is recorded [{param_id}]")
def test_verify_models_rejects_unreviewed_weights(model_service, selected, listed):
    model_service.catalog.json.return_value = {"models": [{"name": n, "digest": d} for n, d in listed.items()]}

    with pytest.raises(ValueError, match=f"^Installed model differs from reviewed weights: {selected[0]}$"):
        environment.verify_models(model_service.lock, selected, model_service.output)
    assert not model_service.output.exists()


@pytest.mark.parametrize("reply", ["catalog", "version"])
@title("HTTP errors from Ollama are raised and nothing is recorded [{reply}]")
def test_verify_models_raises_http_errors(model_service, reply):
    getattr(model_service, reply).raise_for_status.side_effect = requests.HTTPError(reply)

    with pytest.raises(requests.HTTPError, match=reply):
        environment.verify_models(model_service.lock, list(LOCK), model_service.output)
    assert not model_service.output.exists()


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["ci-environment", *map(str, args)])
    return environment.main()


@title("Every operation is refused outside a hosted runner")
def test_cli_requires_hosted_runner(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    with pytest.raises(ValueError, match="disposable GitHub-hosted runner"):
        run_cli(monkeypatch, "wait-ollama")


@pytest.fixture
def operations(monkeypatch, hosted):
    calls = Mock()
    for name in ("bootstrap", "wait_for_service", "verify_models"):
        monkeypatch.setattr(environment, name, getattr(calls, name))
    return calls


@title("The bootstrap operation bootstraps the resolved repository root")
def test_cli_bootstrap(operations, monkeypatch, tmp_path):
    assert run_cli(monkeypatch, "bootstrap", "--root", tmp_path) == 0

    assert operations.mock_calls == [call.bootstrap(tmp_path.resolve())]


@title("The repository root defaults to the parent of the working directory")
def test_cli_default_root(operations, monkeypatch, tmp_path):
    (tmp_path / "automation").mkdir()
    monkeypatch.chdir(tmp_path / "automation")

    run_cli(monkeypatch, "bootstrap")

    assert operations.mock_calls == [call.bootstrap(tmp_path.resolve())]


@title("The wait-ollama operation waits for the local Ollama service")
def test_cli_wait_ollama(operations, monkeypatch):
    assert run_cli(monkeypatch, "wait-ollama") == 0

    assert operations.mock_calls == [call.wait_for_service("http://127.0.0.1:11434")]


@title("The verify-models operation checks the selected models against the reviewed lock")
def test_cli_verify_models(operations, monkeypatch, tmp_path):
    code = run_cli(
            monkeypatch, "verify-models", "--root", tmp_path, "--model", "qwen3.5:4b", "--model", "bge-m3:567m",
            "--output", tmp_path / "env.json")

    assert code == 0
    assert operations.mock_calls == [
            call.verify_models(
            tmp_path / "config/ci-models.json", ["qwen3.5:4b", "bge-m3:567m"], tmp_path / "env.json")]


@pytest.mark.parametrize(
        "arguments", [
        pytest.param(["--output", "env.json"], id="no-model"),
        pytest.param(["--model", "qwen3.5:4b"], id="no-output")])
@title("Model verification requires models and an output [{param_id}]")
def test_cli_verify_models_requires_arguments(operations, monkeypatch, capsys, arguments):
    with pytest.raises(SystemExit) as exit:
        run_cli(monkeypatch, "verify-models", *arguments)

    assert exit.value.code == 2
    assert "Model verification requires --model and --output" in capsys.readouterr().err
    operations.verify_models.assert_not_called()


@title("An unknown operation is a usage error")
def test_cli_rejects_unknown_operation(operations, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exit:
        run_cli(monkeypatch, "deploy")

    assert exit.value.code == 2
    assert "invalid choice: 'deploy'" in capsys.readouterr().err


CAPTURE_SCRIPT = r"""
const Module = require('node:module');
const fs = require('node:fs');
const load = Module._load;
class Ollama { chat() { return 'original-result'; } }
Module._load = function(name, ...rest) { return name === 'ollama' ? { Ollama } : load.call(this, name, ...rest); };
process.env.LLM_TESTKIT_CAPTURE_DIR = process.argv[2];
require(process.argv[1]);
const Client = require('ollama').Ollama;
const result = new Client().chat({messages: [{role: 'system', content: '[LLM_TESTKIT_CAPTURE:' + 'a'.repeat(32) + ']'}]});
const saved = fs.readdirSync(process.argv[2])[0];
console.log(JSON.stringify({mode: fs.statSync(process.argv[2] + '/' + saved).mode & 511, result}));
"""


@pytest.mark.parametrize(("actions", "shared", "mode"), [
        pytest.param("true", "true", 0o644, id="enabled-in-ci"),
        pytest.param("false", "true", 0o600, id="outside-ci"),
        pytest.param("true", "false", 0o600, id="not-enabled")])
@title("Context capture shares read access only in the explicitly enabled CI mode [{param_id}]")
def test_capture_permissions(tmp_path, actions, shared, mode):
    node = shutil.which("node")
    assert node, "Node.js is required to verify CI capture permissions"
    preload = AUTOMATION_ROOT / "src/llm_testkit/observation/ollama-preload.cjs"
    variables = {**os.environ, "GITHUB_ACTIONS": actions, "LLM_TESTKIT_CAPTURE_SHARED_READ": shared}

    result = subprocess.run(
            [node, "-e", CAPTURE_SCRIPT, str(preload), str(tmp_path)], env=variables, check=True, text=True,
            capture_output=True)

    assert json.loads(result.stdout) == {"mode": mode, "result": "original-result"}
