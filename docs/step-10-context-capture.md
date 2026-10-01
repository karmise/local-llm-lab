# Step 10: Actual Model Context Capture

## Purpose and layers

Collect evaluation inputs before introducing RAGAS or another model judge.
The application preload observes the Ollama SDK's `chat` call after AnythingLLM
has compressed its messages. The Python observation layer extracts document
passages from those messages. A fixture combines the observation with the
scenario's reference answer and the completed application response. Tests keep
using the centralized assertions module.

No proxy, new Python dependency or evaluator model is required. The hook returns
the original SDK result and forwards the original request object unchanged.
It captures marked requests only, without authorization headers or unrelated UI
conversations. A capture write error fails the marked request rather than
silently providing fabricated evaluation inputs.

## Enable and run one scenario

Run from the repository root with Docker Desktop and Ollama running:

```bash
mkdir -p .runtime/ollama-capture
docker compose -f compose.yaml -f compose.capture.yaml up -d anythingllm
cd automation
.venv/bin/python -m pytest tests/test_rag.py -k paid_leave \
  --rag-model qwen3.5:4b --capture-rag -v \
  --junitxml=reports/step-10-capture.xml
```

Enabling the overlay recreates the container and preserves its existing storage
volume. Wait for the application to become available before starting the test.
To disable observation, run from the repository root:

```bash
docker compose -f compose.yaml up -d anythingllm
```

`./lab.sh start` also applies the ordinary configuration. `./lab.sh restart`
restarts the current container configuration. Running with `--capture-rag`
without enabling the overlay fails because the expected capture is absent.
Ordinary test runs do not add capture markers or produce capture files.

## Artifacts and interpretation

Raw marked SDK requests are saved under `.runtime/ollama-capture/`.
Evaluation samples are saved under `automation/reports/rag-samples/`, with one
unique file per test invocation. Both directories are excluded from Git. Files
are created with owner-only permissions and are not overwritten.

Each sample includes:

- `user_input`: the exact scenario question, checked against the observed request.
- `retrieved_contexts`: document passages extracted from actual system messages.
- `response`: the completed final answer, with thinking tags and Markdown emphasis
  removed and whitespace normalized by the existing answer assertion.
- `reference`: the scenario's expected answer; never sent to the model.
- `observation`: the complete SDK request, including model, messages and options.
- `response_sources`: application citation metadata, kept separately from context.
- `metadata`: model digest, policy checksum, workspace configuration and repetition.

These are preparation inputs, not computed RAGAS scores. Keep the original
observation when adapting samples to an evaluator's schema.

## Scope and limitations

This adapter targets the pinned AnythingLLM 1.16.2 Ollama connector and text-only,
non-streaming queries with history disabled. The observed boundary is the SDK
call, not a network packet capture. In this connector the text messages are
passed directly to Ollama; image encoding and server-side tokenization are not
observed. The hook forwards streaming calls unchanged, but the Python sample
adapter deliberately rejects them until streaming response collection is added.

An opt-in workspace system prompt contains a unique observation marker. This is
a real prompt difference from ordinary runs and is recorded in configuration
metadata. Captured runs should be compared with other captured runs. Marker
loss during compression, extra model calls, history, malformed context boundaries
or a different question/model cause collection to fail.

The parser preserves each context passage verbatim and excludes the system
instructions. It validates numbered `[CONTEXT n]` boundaries against the pinned
connector format. Treat a future application upgrade as requiring adapter review.
The fixture writes a sample before scenario-specific fact checks, so a completed
but incorrect answer can still be inspected. Transport errors or incomplete
answers fail without creating a completed evaluation sample; the raw request may
still be available locally.

## Verification

Unit coverage checks passage extraction, mismatched questions/models, unexpected
history, missing markers, malformed or empty context, exclusive artifact writes,
and hook transparency for request objects, return values, streams and errors.
Hook verification uses Node.js and a fake SDK without any model requests.

On 2026-10-01, all 33 framework unit checks passed. One captured paid-leave
scenario on Qwen3.5 4B passed in 35.85 seconds. The observed request contained
two messages (system/user), two document passages (1116 and 253 characters),
temperature 0.1 and context limit 8192. Both expected policy facts were present
in the actual context. The application was returned to its ordinary
configuration after verification. Raw artifacts remain local and ignored.
