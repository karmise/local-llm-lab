# Environment Decisions

Recorded on 2026-10-01; user timezone: Asia/Bishkek.

## Platform

- macOS 27.0.1, build 26A434, arm64, Apple M3 Pro, 18 GiB physical memory.
- Approximately 252 GiB free storage before installation.
- Existing Docker Desktop 4.45.0, Engine 28.3.3 and Compose 2.39.2 were retained.
- Docker VM memory: approximately 7.65 GiB. Models run on the host.
- AnythingLLM v1.16.2 uses a native linux/arm64 image.
- Ollama 0.35.0 was installed through Homebrew.

Homebrew updated ca-certificates, openssl@3, readline and xz, and installed
mpdecimal, sqlite, python@3.14, mlx and mlx-c. A pkgconf reinstall warning related
to Xcode did not prevent Ollama installation. Xcode and system Python settings
were not changed. The automation framework uses its separate Python 3.12 environment.

## Architecture and models

AnythingLLM provides the UI, API and RAG pipeline in Docker. Native Ollama uses
Apple Silicon GPU access. Qwen2.5 7B Q4_K_M was the initial baseline; Qwen3.5 4B
Q4_K_M was subsequently installed and selected as the default. Both are retained
for comparisons. The context limit is 8192 tokens. BGE-M3 provides multilingual
embeddings; LanceDB avoids an additional database service.

The initial workspace settings are query mode, temperature 0.1, history 0,
topN 4 and similarity threshold 0.25. These are starting settings, not validated
optimal quality thresholds. History remains stored in the application while
being excluded from new model requests.

Qwen3.5 supports thinking. The AnythingLLM v1.16.2 Ollama connector wraps returned
thinking in `<think>` tags without supplying an explicit `think` request option.
Comparisons must record the thinking mode and separate reasoning from the final
answer. Generation model switching preserves embeddings and the vector index.

## Persistence and access

Compose project: `denis-llm-lab`; storage volume:
`denis-llm-lab_anythingllm-storage`. Application access is bound to 127.0.0.1:3001;
Ollama uses port 11434. Container access via `host.docker.internal` was verified.
Paid model APIs are not configured and AnythingLLM telemetry is disabled. Model
metadata downloads do not establish complete network isolation.

Initial setup used the application's internal UI API. Subsequent framework work
will use documented developer API `/api/v1` and keep API keys outside Git.
Internal UI endpoints are not assumed to be a public API contract.

## Verification limits

Initial smoke checks established connectivity, document retrieval and persistence.
Statistical quality evaluation, RAGAS, prompt injection coverage, drift monitoring
and CI remain future work. A consistent model-dependent failure is distinct from
a flaky result across identical repeated configurations.

The repository now uses English configuration and test data. Earlier localized
raw responses and screenshots were archived outside the project rather than
translated into purported English execution results. Existing application data
in the Docker volume is not automatically migrated by editing project files.
