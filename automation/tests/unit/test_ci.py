import json

import pytest

from llm_testkit.ci import environment
from llm_testkit.ci.benchmark import execution_context, inputs
from llm_testkit.reporting.steps import title
from test_support.assertions import errors, values
from test_support.data.benchmark import ROOT
from test_support.data.ci import INVALID_IDENTITY, MODEL_NAMES, PRESETS, REVISION
from test_support.fixtures.unit_benchmark import benchmark_data as benchmark_data
from test_support.fixtures.unit_ci import bootstrap_service as bootstrap_service
from test_support.fixtures.unit_ci import ci_run as ci_run
from test_support.fixtures.unit_ci import hosted_environment as hosted_environment
from test_support.fixtures.unit_ci import model_service as model_service

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("profile,models,generations,maximum_calls", PRESETS)
@title("CI presets declare a bounded complete matrix before model calls [{param_id}]")
def test_ci_presets(profile, models, generations, maximum_calls):
    _, _, plan = inputs(ROOT, profile, models)
    values.equal(len(plan.case_ids) * len(plan.models), generations)
    values.equal(plan.maximum_calls, maximum_calls)


@title("Independent CI gate validates fresh evidence without application or judge calls")
def test_saved_ci_gate(ci_run):
    values.equal(ci_run.validate()["status"], "checks_passed")


@pytest.mark.parametrize("change", INVALID_IDENTITY)
@title("CI rejects unrelated revisions, edited expectations and changed model or test provenance [{param_id}]")
def test_changed_ci_evidence(ci_run, change):
    ci_run.change(change)
    errors.rejects(ci_run.validate, expected=(ValueError, AssertionError))


@title("Generation errors retain a failing CI verdict even when a saved summary claimed success")
def test_failed_ci_generation(ci_run):
    ci_run.fail_generation()
    values.equal(ci_run.validate()["status"], "error")


@title("CI refuses abbreviated revision identities")
def test_invalid_revision():
    errors.rejects(lambda: execution_context(ROOT, REVISION[:7], "smoke", "primary"), expected=ValueError)


@title("Application bootstrap refuses the ordinary developer environment")
def test_bootstrap_refuses_local_environment(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    errors.rejects(environment.require_hosted_environment, expected=ValueError, match="disposable")


@title("Disposable CI API key is verified and saved privately without entering reports")
def test_bootstrap_key(bootstrap_service, capsys):
    root, secret, _ = bootstrap_service
    environment.bootstrap(root)
    path = root / ".runtime/anythingllm-api-key"
    values.equal(path.read_text(), secret)
    values.equal(path.stat().st_mode & 0o777, 0o600)
    values.contains(capsys.readouterr().out, "::add-mask::")
    errors.rejects(lambda: environment.bootstrap(root), expected=ValueError, match="existing API key")


@title("CI records verified generation and embedding weights before generation")
def test_model_weights(model_service):
    lock, output, _ = model_service
    environment.verify_models(lock, list(MODEL_NAMES), output)
    values.equal(set(json.loads(output.read_text())["models"]), set(MODEL_NAMES))


@title("Changed downloaded model tags fail preflight rather than silently updating the experiment")
def test_changed_model_weights(model_service):
    lock, output, catalog = model_service
    catalog.json.return_value["models"][0]["digest"] = "changed"
    errors.rejects(lambda: environment.verify_models(lock, list(MODEL_NAMES), output), expected=ValueError)
    values.falsy(output.exists())
