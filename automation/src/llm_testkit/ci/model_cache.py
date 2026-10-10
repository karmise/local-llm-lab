"""Install reviewed Ollama manifests and content-addressed blobs in disposable CI."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import requests

from llm_testkit.ci.environment import require_hosted_environment


def reviewed_manifest(root: Path, model: str) -> tuple[bytes, list[dict]]:
    lock = json.loads((root / "config/ci-models.json").read_text())
    if model not in lock or not re.fullmatch(r"[a-z0-9_.-]+:[a-zA-Z0-9_.-]+", model):
        raise ValueError("Select a reviewed library model")
    raw = (root / "config/ci-model-manifests" / (model.replace(":", "-") + ".json")).read_bytes()
    if hashlib.sha256(raw).hexdigest() != lock[model]:
        raise ValueError(f"Reviewed model manifest checksum mismatch: {model}")
    value = json.loads(raw)
    layers = [value["config"], *value["layers"]]
    if value.get("schemaVersion") != 2 or not value["layers"]:
        raise ValueError("Unsupported model manifest")
    for layer in layers:
        if (not re.fullmatch(r"sha256:[a-f0-9]{64}", layer.get("digest", "")) or type(layer.get("size")) is not int
                    or not 0 < layer["size"] <= 10_000_000_000):
            raise ValueError("Invalid reviewed model blob")
    return raw, layers


def download_blob(session: requests.Session, model: str, layer: dict, directory: Path) -> None:
    digest = layer["digest"]
    target = directory / digest.replace(":", "-")
    temporary = target.with_suffix(".partial")
    checksum = hashlib.sha256()
    size = 0
    print(f"Downloading reviewed {model} blob {digest[7:19]} ({layer['size']} bytes)", flush=True)
    url = f"https://registry.ollama.ai/v2/library/{model.split(':')[0]}/blobs/{digest}"
    try:
        with session.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            with temporary.open("xb") as file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    size += len(chunk)
                    if size > layer["size"]:
                        raise ValueError("Model blob exceeds its reviewed size")
                    checksum.update(chunk)
                    file.write(chunk)
        if size != layer["size"] or checksum.hexdigest() != digest[7:]:
            raise ValueError("Downloaded model blob checksum/size mismatch")
        # No server is started yet, so the freshly verified blob is not read mid-write.
        if target.exists():
            raise ValueError("Refusing to replace an existing model blob")
        temporary.rename(target)
    finally:
        temporary.unlink(missing_ok=True)


def install_models(root: Path, selected: list[str]) -> None:
    require_hosted_environment()
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select unique reviewed models")
    definitions = [(model, *reviewed_manifest(root, model)) for model in selected]
    cache = root / ".runtime/ollama-models"
    cache.mkdir(parents=True, exist_ok=False)
    blobs = cache / "blobs"
    blobs.mkdir()
    installed = set()
    with requests.Session() as session:
        for model, raw, layers in definitions:
            for layer in layers:
                if layer["digest"] not in installed:
                    download_blob(session, model, layer, blobs)
                    installed.add(layer["digest"])
            name, tag = model.split(":")
            manifest = cache / "manifests/registry.ollama.ai/library" / name / tag
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_bytes(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd().parent)
    parser.add_argument("--model", action="append", dest="models", required=True)
    args = parser.parse_args()
    install_models(args.root.resolve(), args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
