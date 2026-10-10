"""CI scenarios supply prepared reports and service clients without network access."""

import json
import shutil
from unittest.mock import Mock

import pytest

from llm_testkit.ci import environment
from llm_testkit.evaluation import benchmark_runner as runner
from test_support.builders.benchmark import make_calibrate_stub, make_judge_model_catalog
from test_support.builders.ci import ci_generator, create_run
from test_support.data.benchmark import ROOT


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
