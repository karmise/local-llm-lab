"""Bootstrap only a fresh GitHub-hosted application; never export its credentials."""

import argparse
import json
import os
import time
from pathlib import Path

import requests

from llm_testkit.clients.anythingllm_client import AnythingLLMClient
from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.core.http_client import HttpClient
from llm_testkit.observation.evaluation_sample import write_sample


def require_hosted_environment() -> None:
    if os.getenv("GITHUB_ACTIONS") != "true" or os.getenv("RUNNER_ENVIRONMENT") != "github-hosted":
        raise ValueError("Application bootstrap is restricted to a disposable GitHub-hosted runner")


def wait_for_service(url: str, *, attempts: int = 90) -> None:
    with HttpClient(url, 5) as client:
        for _ in range(attempts):
            try:
                if client.request("GET", "/api/ping" if url.endswith(":3001") else "/api/version").status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
    raise ValueError(f"Service did not become ready: {url}")


def bootstrap(root: Path) -> None:
    require_hosted_environment()
    runtime = root / ".runtime"
    key_path = runtime / "anythingllm-api-key"
    if key_path.exists():
        raise ValueError("Refusing to bootstrap an existing API key")
    runtime.mkdir(mode=0o700, exist_ok=True)
    wait_for_service("http://127.0.0.1:3001")
    with HttpClient("http://127.0.0.1:3001", 10) as http:
        response = http.request("POST", "/api/system/generate-api-key", json={"name": "disposable-ci"})
        response.raise_for_status()
        payload = response.json()
        secret = (payload.get("apiKey") or {}).get("secret")
        if payload.get("error") or not isinstance(secret, str) or not secret.strip():
            raise ValueError("Disposable API key creation failed")
        # Mask before any subsequent operation; the key is never an artifact or job output.
        print(f"::add-mask::{secret}", flush=True)
        verification = AnythingLLMClient(http, api_key=secret).verify_authentication()
        verification.raise_for_status()
        if verification.json().get("authenticated") is not True:
            raise ValueError("Disposable API key was not authenticated")
    descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as key_file:
        key_file.write(secret)


def verify_models(lock_path: Path, selected: list[str], output: Path) -> None:
    expected = json.loads(lock_path.read_text())
    with HttpClient("http://127.0.0.1:11434", 10) as http:
        response = OllamaClient(http).list_models()
        response.raise_for_status()
        catalog = {entry["name"]: entry["digest"] for entry in response.json()["models"]}
        for model in selected:
            if model not in expected or catalog.get(model) != expected[model]:
                raise ValueError(f"Installed model differs from reviewed weights: {model}")
        version = http.request("GET", "/api/version")
        version.raise_for_status()
    write_sample(output, {"ollama_version": version.json()["version"], "models": {m: catalog[m] for m in selected}})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("bootstrap", "wait-ollama", "verify-models"))
    parser.add_argument("--root", type=Path, default=Path.cwd().parent)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require_hosted_environment()
    if args.operation == "bootstrap":
        bootstrap(args.root.resolve())
    elif args.operation == "wait-ollama":
        wait_for_service("http://127.0.0.1:11434")
    else:
        if not args.models or args.output is None:
            parser.error("Model verification requires --model and --output")
        verify_models(args.root / "config/ci-models.json", args.models, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
