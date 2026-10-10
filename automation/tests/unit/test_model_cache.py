"""Model cache: reviewed Ollama manifests and checksum-verified, content-addressed blobs for disposable CI."""

import hashlib
import json
import sys
from unittest.mock import MagicMock, Mock, call

import pytest
import requests

from llm_testkit.ci import model_cache
from llm_testkit.ci.model_cache import download_blob, install_models, reviewed_manifest
from llm_testkit.reporting.steps import title
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit

REPOSITORY = AUTOMATION_ROOT.parent
BLOB = b"reviewed fake model bytes"


def blob_layer(content: bytes = BLOB) -> dict:
    return {"digest": "sha256:" + hashlib.sha256(content).hexdigest(), "size": len(content)}


@pytest.mark.parametrize("model", ["qwen3.5:4b", "qwen2.5:7b", "bge-m3:567m"])
@title("Checked-in model manifests match the reviewed digest lock [{model}]")
def test_reviewed_manifest(model):
    raw, layers = reviewed_manifest(REPOSITORY, model)

    lock = json.loads((REPOSITORY / "config/ci-models.json").read_text())
    manifest = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == lock[model]
    assert layers == [manifest["config"], *manifest["layers"]]


def write_repository(tmp_path, model="test:1b", manifest=None, *, lock_digest=None) -> object:
    """A repository whose lock reviews one model manifest (by default a valid one with one layer)."""
    manifest = {
            "schemaVersion": 2,
            "config": blob_layer(b"config"),
            "layers": [blob_layer()]} if manifest is None else manifest
    raw = json.dumps(manifest).encode()
    (tmp_path / "config/ci-model-manifests").mkdir(parents=True)
    (tmp_path / "config/ci-model-manifests" / (model.replace(":", "-") + ".json")).write_bytes(raw)
    (tmp_path / "config/ci-models.json").write_text(json.dumps({model: lock_digest or hashlib.sha256(raw).hexdigest()}))
    return tmp_path


@title("A valid reviewed manifest returns its bytes and the config and model layers")
def test_reviewed_manifest_accepts_valid(tmp_path):
    raw, layers = reviewed_manifest(write_repository(tmp_path), "test:1b")

    assert layers == [blob_layer(b"config"), blob_layer()]
    assert json.loads(raw)["schemaVersion"] == 2


@pytest.mark.parametrize(
        "model", [
        pytest.param("other:1b", id="not-in-lock"),
        pytest.param("Test:1b", id="uppercase-name"),
        pytest.param("test", id="no-tag"),
        pytest.param("../test:1b", id="path")])
@title("Only a reviewed library model name can be selected [{param_id}]")
def test_reviewed_manifest_rejects_unreviewed_model(tmp_path, model):
    root = write_repository(tmp_path)
    lock = json.loads((root / "config/ci-models.json").read_text())
    (root / "config/ci-models.json").write_text(json.dumps(lock | {"Test:1b": "x", "test": "x", "../test:1b": "x"}))

    with pytest.raises(ValueError, match="Select a reviewed library model"):
        reviewed_manifest(root, model)


@title("A tag may use upper-case letters, digits, dots, underscores and hyphens")
def test_reviewed_manifest_accepts_tag_characters(tmp_path):
    raw, _ = reviewed_manifest(write_repository(tmp_path, "test-model_v2.1:Q4_K-M.0"), "test-model_v2.1:Q4_K-M.0")

    assert raw


@title("A manifest that differs from its locked checksum is rejected")
def test_reviewed_manifest_rejects_changed_manifest(tmp_path):
    root = write_repository(tmp_path, lock_digest="0" * 64)

    with pytest.raises(ValueError, match=r"^Reviewed model manifest checksum mismatch: test:1b$"):
        reviewed_manifest(root, "test:1b")


@pytest.mark.parametrize(("manifest", "message"), [
        pytest.param({
        "schemaVersion": 1,
        "config": blob_layer(),
        "layers": [blob_layer()]}, "Unsupported model manifest", id="schema-one"),
        pytest.param({
        "config": blob_layer(),
        "layers": [blob_layer()]}, "Unsupported model manifest", id="no-schema"),
        pytest.param({
        "schemaVersion": 2,
        "config": blob_layer(),
        "layers": []}, "Unsupported model manifest", id="no-layers"),
        pytest.param({
        "schemaVersion": 2,
        "config": {
        "size": 1},
        "layers": [blob_layer()]}, "Invalid reviewed model blob", id="config-without-digest"),
        pytest.param({
        "schemaVersion": 2,
        "config": blob_layer(),
        "layers": [{
        "digest": "md5:" + "a" * 64,
        "size": 1}]}, "Invalid reviewed model blob", id="not-sha256"),
        pytest.param({
        "schemaVersion": 2,
        "config": blob_layer(),
        "layers": [{
        "digest": "sha256:" + "A" * 64,
        "size": 1}]}, "Invalid reviewed model blob", id="uppercase-digest"),
        pytest.param({
        "schemaVersion": 2,
        "config": blob_layer(),
        "layers": [blob_layer() | {
        "size": "25"}]}, "Invalid reviewed model blob", id="string-size"),
        pytest.param({
        "schemaVersion": 2,
        "config": blob_layer(),
        "layers": [blob_layer() | {
        "size": 0}]}, "Invalid reviewed model blob", id="empty-blob"),
        pytest.param({
        "schemaVersion": 2,
        "config": blob_layer(),
        "layers": [blob_layer() | {
        "size": 10_000_000_001}]}, "Invalid reviewed model blob", id="oversized-blob")])
@title("A reviewed manifest with an unsupported schema or invalid blob is rejected [{param_id}]")
def test_reviewed_manifest_rejects_invalid_manifest(tmp_path, manifest, message):
    root = write_repository(tmp_path, manifest=manifest)

    with pytest.raises(ValueError, match=message):
        reviewed_manifest(root, "test:1b")


@title("A blob of exactly the largest allowed size is accepted")
def test_reviewed_manifest_accepts_largest_blob(tmp_path):
    layer = blob_layer() | {"size": 10_000_000_000}
    root = write_repository(tmp_path, manifest={"schemaVersion": 2, "config": blob_layer(), "layers": [layer]})

    assert reviewed_manifest(root, "test:1b")[1][1] == layer


@pytest.fixture
def registry():
    """A registry session streaming the reviewed blob."""
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_content.return_value = [BLOB]
    session = Mock()
    session.get.return_value = response
    return session


@title("A complete, checksum-verified blob is published under its digest")
def test_download_blob_publishes_verified_blob(tmp_path, registry, capsys):
    layer = blob_layer()

    download_blob(registry, "qwen3.5:4b", layer, tmp_path)

    target = tmp_path / layer["digest"].replace(":", "-")
    assert target.read_bytes() == BLOB
    assert list(tmp_path.iterdir()) == [target]
    registry.get.assert_called_once_with(
            f"https://registry.ollama.ai/v2/library/qwen3.5/blobs/{layer['digest']}", stream=True, timeout=(10, 60))
    registry.get.return_value.iter_content.assert_called_once_with(chunk_size=1024 * 1024)
    assert capsys.readouterr().out == (
            f"Downloading reviewed qwen3.5:4b blob {layer['digest'][7:19]} ({len(BLOB)} bytes)\n")


@title("A blob streamed in several chunks is verified as a whole")
def test_download_blob_joins_chunks(tmp_path, registry):
    registry.get.return_value.iter_content.return_value = [BLOB[:5], BLOB[5:]]

    download_blob(registry, "qwen3.5:4b", blob_layer(), tmp_path)

    assert (tmp_path / blob_layer()["digest"].replace(":", "-")).read_bytes() == BLOB


@pytest.mark.parametrize(("chunks", "message"), [
        pytest.param([b"wrong"], "Downloaded model blob checksum/size mismatch", id="truncated"),
        pytest.param([b"x" * len(BLOB)], "Downloaded model blob checksum/size mismatch", id="changed"),
        pytest.param([BLOB, b"extra"], "Model blob exceeds its reviewed size", id="oversized")])
@title("Truncated, changed and oversized downloads cannot become installed weights [{param_id}]")
def test_download_blob_rejects_invalid_content(tmp_path, registry, chunks, message):
    registry.get.return_value.iter_content.return_value = chunks

    with pytest.raises(ValueError, match=message):
        download_blob(registry, "qwen3.5:4b", blob_layer(), tmp_path)
    assert list(tmp_path.iterdir()) == []


@title("An oversized download stops reading as soon as it exceeds the reviewed size")
def test_download_blob_stops_oversized_stream(tmp_path, registry):
    read = []

    def chunks():
        for chunk in (BLOB, b"x", b"never read"):
            read.append(chunk)
            yield chunk

    registry.get.return_value.iter_content.return_value = chunks()

    with pytest.raises(ValueError, match="exceeds its reviewed size"):
        download_blob(registry, "qwen3.5:4b", blob_layer(), tmp_path)
    assert read == [BLOB, b"x"]


@title("An HTTP error leaves no partial file")
def test_download_blob_raises_http_error(tmp_path, registry):
    registry.get.return_value.raise_for_status.side_effect = requests.HTTPError("404")

    with pytest.raises(requests.HTTPError):
        download_blob(registry, "qwen3.5:4b", blob_layer(), tmp_path)
    assert list(tmp_path.iterdir()) == []


@title("An installed blob is never replaced")
def test_download_blob_refuses_existing_blob(tmp_path, registry):
    download_blob(registry, "qwen3.5:4b", blob_layer(), tmp_path)

    with pytest.raises(ValueError, match="Refusing to replace an existing model blob"):
        download_blob(registry, "qwen3.5:4b", blob_layer(), tmp_path)
    assert [path.read_bytes() for path in tmp_path.iterdir()] == [BLOB]


@title("A leftover partial download is not overwritten")
def test_download_blob_refuses_leftover_partial(tmp_path, registry):
    partial = tmp_path / (blob_layer()["digest"].replace(":", "-") + ".partial")
    partial.write_bytes(b"other download")

    with pytest.raises(FileExistsError):
        download_blob(registry, "qwen3.5:4b", blob_layer(), tmp_path)


@pytest.fixture
def hosted(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")


@pytest.fixture
def downloads(monkeypatch):
    """Record downloads instead of contacting the registry."""
    download = Mock(side_effect=lambda session, model, layer, directory: (directory / layer["digest"]).touch())
    monkeypatch.setattr(model_cache, "download_blob", download)
    monkeypatch.setattr(model_cache.requests, "Session", MagicMock())
    return download


def two_model_repository(tmp_path):
    shared = blob_layer(b"shared license")
    manifests = {
            "first:1b": {
            "schemaVersion": 2,
            "config": blob_layer(b"config-1"),
            "layers": [blob_layer(b"weights-1"), shared]},
            "first:2b": {
            "schemaVersion": 2,
            "config": blob_layer(b"config-2"),
            "layers": [shared]}}
    (tmp_path / "config/ci-model-manifests").mkdir(parents=True)
    lock = {}
    for model, manifest in manifests.items():
        raw = json.dumps(manifest).encode()
        (tmp_path / "config/ci-model-manifests" / (model.replace(":", "-") + ".json")).write_bytes(raw)
        lock[model] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "config/ci-models.json").write_text(json.dumps(lock))
    return manifests


@title("Installation downloads each blob once and writes the reviewed manifests in Ollama's layout")
def test_install_models(tmp_path, hosted, downloads):
    manifests = two_model_repository(tmp_path)

    install_models(tmp_path, ["first:1b", "first:2b"])

    cache = tmp_path / ".runtime/ollama-models"
    assert [(c.args[1], c.args[2]["digest"]) for c in downloads.call_args_list] == [
            ("first:1b", manifests["first:1b"]["config"]["digest"]),
            ("first:1b", manifests["first:1b"]["layers"][0]["digest"]),
            ("first:1b", manifests["first:1b"]["layers"][1]["digest"]),
            ("first:2b", manifests["first:2b"]["config"]["digest"])]
    session = model_cache.requests.Session.return_value.__enter__.return_value
    assert {(c.args[0], c.args[3]) for c in downloads.call_args_list} == {(session, cache / "blobs")}
    for model, manifest in manifests.items():
        name, tag = model.split(":")
        assert json.loads((cache / "manifests/registry.ollama.ai/library" / name / tag).read_text()) == manifest


@pytest.mark.parametrize("selected", [pytest.param([], id="none"), pytest.param(["first:1b"] * 2, id="duplicate")])
@title("Installation requires a unique selection of reviewed models [{param_id}]")
def test_install_models_rejects_selection(tmp_path, hosted, downloads, selected):
    two_model_repository(tmp_path)

    with pytest.raises(ValueError, match="Select unique reviewed models"):
        install_models(tmp_path, selected)
    downloads.assert_not_called()


@title("Every manifest is verified before any download starts")
def test_install_models_verifies_all_manifests_first(tmp_path, hosted, downloads):
    two_model_repository(tmp_path)

    with pytest.raises(ValueError, match="Select a reviewed library model"):
        install_models(tmp_path, ["first:1b", "unreviewed:1b"])
    downloads.assert_not_called()
    assert not (tmp_path / ".runtime").exists()


@title("Installation refuses to reuse an existing model cache")
def test_install_models_refuses_existing_cache(tmp_path, hosted, downloads):
    two_model_repository(tmp_path)
    (tmp_path / ".runtime/ollama-models").mkdir(parents=True)

    with pytest.raises(FileExistsError):
        install_models(tmp_path, ["first:1b"])
    downloads.assert_not_called()


@title("Installation is refused outside a disposable hosted runner")
def test_install_models_requires_hosted_runner(tmp_path, monkeypatch, downloads):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    two_model_repository(tmp_path)

    with pytest.raises(ValueError, match="disposable GitHub-hosted runner"):
        install_models(tmp_path, ["first:1b"])
    downloads.assert_not_called()


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["model-cache", *map(str, args)])
    return model_cache.main()


@title("The command installs the selected models into the resolved repository root")
def test_cli_installs_models(tmp_path, monkeypatch):
    install = Mock()
    monkeypatch.setattr(model_cache, "install_models", install)

    assert run_cli(monkeypatch, "--root", tmp_path, "--model", "a:1b", "--model", "b:2b") == 0

    assert install.mock_calls == [call(tmp_path.resolve(), ["a:1b", "b:2b"])]


@title("The repository root defaults to the parent of the working directory")
def test_cli_default_root(tmp_path, monkeypatch):
    install = Mock()
    monkeypatch.setattr(model_cache, "install_models", install)
    (tmp_path / "automation").mkdir()
    monkeypatch.chdir(tmp_path / "automation")

    run_cli(monkeypatch, "--model", "a:1b")

    assert install.call_args.args[0] == tmp_path.resolve()


@title("The command requires at least one model")
def test_cli_requires_model(monkeypatch):
    with pytest.raises(SystemExit) as exit:
        run_cli(monkeypatch)

    assert exit.value.code == 2
