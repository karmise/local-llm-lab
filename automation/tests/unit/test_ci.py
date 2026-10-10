import hashlib
import json

import pytest

from llm_testkit.ci import environment
from llm_testkit.ci.benchmark import execution_context, inputs
from llm_testkit.ci.model_cache import reviewed_manifest
from llm_testkit.evaluation.benchmark_runner import generate_sample
from llm_testkit.reporting.redaction import redact_files
from llm_testkit.reporting.steps import title
from test_support.assertions import errors, values
from test_support.builders.ci import reject_authentication
from test_support.data.benchmark import ROOT
from test_support.data.ci import (
        BLOB_BYTES, CAPTURE_PERMISSIONS, INVALID_BLOBS, INVALID_IDENTITY, MODEL_NAMES, PRESETS, REVIEWED_MODELS,
        REVISION)
from test_support.fixtures.unit_benchmark import benchmark_data as benchmark_data
from test_support.fixtures.unit_ci import bootstrap_service as bootstrap_service
from test_support.fixtures.unit_ci import capture_permissions as capture_permissions
from test_support.fixtures.unit_ci import ci_run as ci_run
from test_support.fixtures.unit_ci import credential_reports as credential_reports
from test_support.fixtures.unit_ci import failed_generation as failed_generation
from test_support.fixtures.unit_ci import hosted_environment as hosted_environment
from test_support.fixtures.unit_ci import model_service as model_service
from test_support.fixtures.unit_ci import reviewed_blob as reviewed_blob

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


@pytest.mark.parametrize("actions,shared,expected_mode", CAPTURE_PERMISSIONS)
@title("Context capture shares read access only for the explicitly enabled disposable CI mode [{param_id}]")
def test_capture_permissions(capture_permissions, actions, shared, expected_mode):
    result = capture_permissions(actions, shared)
    values.equal(result["mode"], expected_mode)
    values.equal(result["result"], "original-result")


@pytest.mark.parametrize("model", REVIEWED_MODELS)
@title("Checked-in immutable model manifests match the reviewed digest lock [{param_id}]")
def test_reviewed_manifest(model):
    raw, layers = reviewed_manifest(ROOT.parent, model)
    lock = json.loads((ROOT.parent / "config/ci-models.json").read_text())
    values.equal(hashlib.sha256(raw).hexdigest(), lock[model])
    values.truthy(layers)


@title("Model installation publishes only a complete checksum-verified blob")
def test_verified_blob(reviewed_blob):
    reviewed_blob.download()
    values.equal(reviewed_blob.saved_bytes(), BLOB_BYTES)
    values.length(reviewed_blob.files(), 1)
    errors.rejects(reviewed_blob.download, expected=ValueError, match="existing")
    values.equal(reviewed_blob.saved_bytes(), BLOB_BYTES)


@pytest.mark.parametrize("content", INVALID_BLOBS)
@title("Truncated, changed and oversized model downloads cannot become installed weights [{param_id}]")
def test_invalid_blob(reviewed_blob, content):
    reviewed_blob.replace_bytes(content)
    errors.rejects(reviewed_blob.download, expected=ValueError)
    values.falsy(reviewed_blob.files())


@title("Archived tracebacks remove repeated credentials while preserving failure diagnostics and sample bytes")
def test_redact_reports(credential_reports):
    directory, secret = credential_reports
    values.equal(redact_files(directory.rglob("*"), secret), 2)
    values.excludes((directory / "generation.log").read_text(), secret)
    values.contains((directory / "generation.log").read_text(), "ReadTimeout")
    values.excludes((directory / "generation.xml").read_text(), secret)
    values.equal((directory / "sample.json").read_text(), '{"answer": "Preserved evidence"}')
    values.equal(redact_files(directory.rglob("*"), secret), 0)


@title("Reports are redacted before JUnit parsing and evidence checksum calculation even on failed generation")
def test_redact_before_generation_hash(failed_generation):
    root, directory, secret, inspected = failed_generation
    errors.rejects(
            lambda: generate_sample(root, directory, "paid_leave", "qwen3.5:4b"), expected=ValueError,
            match="no captured sample")
    content, digest = inspected[0]
    values.excludes(content, secret.encode())
    values.equal(digest, hashlib.sha256(content).hexdigest())
    values.contains(content, b"ReadTimeout")


@title("Absent bootstrap credentials do not modify diagnostic reports")
def test_no_redaction_secret(credential_reports):
    directory, _ = credential_reports
    values.equal(redact_files(directory.rglob("*"), None), 0)


@title("Credential scrub refuses linked evidence rather than modifying an external file")
def test_redaction_symlink(credential_reports):
    directory, secret = credential_reports
    linked = directory / "linked.log"
    linked.symlink_to(directory / "generation.log")
    errors.rejects(lambda: redact_files((linked, ), secret), expected=ValueError, match="symlink")


@title("Failed bootstrap verification retains a private credential for diagnostic redaction")
def test_failed_bootstrap_preserves_redaction_key(bootstrap_service, monkeypatch):
    root, secret, _ = bootstrap_service
    monkeypatch.setattr(environment.AnythingLLMClient, "verify_authentication", reject_authentication)
    errors.rejects(lambda: environment.bootstrap(root), expected=ValueError, match="authentication failed")
    values.equal((root / ".runtime/anythingllm-api-key").read_text(), secret)
    values.equal((root / ".runtime/anythingllm-api-key").stat().st_mode & 0o777, 0o600)
