"""CI scenarios supply prepared reports and service clients without network access."""

import hashlib
import json
import os
import shutil
import subprocess
from unittest.mock import Mock

import pytest

from llm_testkit.ci import environment
from llm_testkit.evaluation import benchmark_runner as runner
from llm_testkit.reporting.junit import read_junit
from test_support.builders.benchmark import make_calibrate_stub, make_judge_model_catalog
from test_support.builders.ci import ReviewedBlob, ci_generator, create_run
from test_support.data.benchmark import ROOT
from test_support.data.ci import BLOB_BYTES, CAPTURE_SCRIPT, REPORT_SECRET, REPORT_TRACE, REPORT_XML


@pytest.fixture
def ci_run(tmp_path, benchmark_data, monkeypatch):
    dataset, gates, plan, _, calibration = benchmark_data
    root = tmp_path / "automation"
    shutil.copytree(ROOT / "test_data", root / "test_data")
    (root / "tests").mkdir()
    shutil.copyfile(ROOT / "tests/test_golden_rag.py", root / "tests/test_golden_rag.py")
    (root.parent / "config").mkdir()
    (root.parent / "config/ci-models.json").write_text(json.dumps({"qwen3.5:4b": "judge-digest"}))
    catalog = Mock(status_code=200)
    catalog.json.return_value = make_judge_model_catalog()
    monkeypatch.setattr(runner.OllamaClient, "list_models", lambda _: catalog)
    monkeypatch.setattr(runner, "calibrate", make_calibrate_stub(calibration))
    monkeypatch.setattr(runner, "generate_sample", ci_generator(dataset, monkeypatch))
    return create_run(root, tmp_path / "run", dataset, gates, plan)


@pytest.fixture
def hosted_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")


@pytest.fixture
def bootstrap_service(tmp_path, monkeypatch, hosted_environment):
    secret = "disposable-unit-test-key"
    response = Mock()
    response.json.return_value = {"apiKey": {"secret": secret}, "error": None}
    http = Mock()
    http.__enter__ = Mock(return_value=http)
    http.__exit__ = Mock(return_value=None)
    http.request.return_value = response
    authentication = Mock()
    authentication.json.return_value = {"authenticated": True}
    monkeypatch.setattr(environment, "HttpClient", Mock(return_value=http))
    monkeypatch.setattr(environment, "wait_for_service", Mock())
    monkeypatch.setattr(environment.AnythingLLMClient, "verify_authentication", Mock(return_value=authentication))
    return tmp_path, secret, http


@pytest.fixture
def model_service(tmp_path, monkeypatch):
    lock = tmp_path / "models.json"
    lock.write_text(json.dumps({"qwen3.5:4b": "qwen-weights", "bge-m3:567m": "embedding-weights"}))
    response = Mock()
    response.json.return_value = {
            "models": [{
            "name": "qwen3.5:4b",
            "digest": "qwen-weights"}, {
            "name": "bge-m3:567m",
            "digest": "embedding-weights"}]}
    http = Mock()
    http.__enter__ = Mock(return_value=http)
    http.__exit__ = Mock(return_value=None)
    http.request.return_value.json.return_value = {"version": "test-version"}
    monkeypatch.setattr(environment, "HttpClient", Mock(return_value=http))
    monkeypatch.setattr(environment.OllamaClient, "list_models", Mock(return_value=response))
    return lock, tmp_path / "environment.json", response


@pytest.fixture
def capture_permissions(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.fail("Node.js is required to verify CI capture permissions")

    def inspect(actions, shared):
        environment = {**os.environ, "GITHUB_ACTIONS": actions, "LLM_TESTKIT_CAPTURE_SHARED_READ": shared}
        result = subprocess.run([
                node, "-e", CAPTURE_SCRIPT,
                str(ROOT / "src/llm_testkit/observation/ollama-preload.cjs"),
                str(tmp_path)], env=environment, check=True, text=True, capture_output=True)
        return json.loads(result.stdout)

    return inspect


@pytest.fixture
def reviewed_blob(tmp_path):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=None)
    response.iter_content.return_value = [BLOB_BYTES]
    session = Mock()
    session.get.return_value = response
    layer = {"digest": "sha256:" + hashlib.sha256(BLOB_BYTES).hexdigest(), "size": len(BLOB_BYTES)}
    return ReviewedBlob(session, response, tmp_path, layer)


@pytest.fixture
def credential_reports(tmp_path):
    directory = tmp_path / "reports"
    directory.mkdir()
    (directory / "generation.log").write_text(REPORT_TRACE)
    (directory / "generation.xml").write_text(REPORT_XML)
    (directory / "sample.json").write_text('{"answer": "Preserved evidence"}')
    return directory, REPORT_SECRET


@pytest.fixture
def failed_generation(credential_reports, monkeypatch):
    directory, secret = credential_reports
    root = directory.parent / "automation"
    (root.parent / ".runtime").mkdir()
    (root.parent / ".runtime/anythingllm-api-key").write_text(secret)
    monkeypatch.delenv("ANYTHINGLLM_API_KEY", raising=False)
    monkeypatch.delenv("ANYTHINGLLM_API_KEY_FILE", raising=False)
    # The subprocess double supplies an actual failing JUnit file and traceback.
    log = (directory / "generation.log").read_text()
    (directory / "generation.log").unlink()

    def execute(*args, stdout, **kwargs):
        stdout.write(log)
        return Mock(returncode=1)

    inspected = []

    def inspect(path):
        result = read_junit(path)
        inspected.append((path.read_bytes(), result.sha256))
        return result

    monkeypatch.setattr(runner.subprocess, "run", execute)
    monkeypatch.setattr(runner, "read_junit", inspect)
    return root, directory, secret, inspected
