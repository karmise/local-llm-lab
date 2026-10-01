# Actual Model Context for RAG Evaluation

Source review date: 2026-10-01. AnythingLLM v1.16.2.

In `server/utils/chats/stream.js`, the reviewed request flow was:

1. Collect `contextTexts` from pinned documents, attachments, retrieval and
   backfilled previous sources.
2. Populate current response `sources` using a different set of passages.
3. Assemble messages through `LLMConnector.compressMessages(...)`, potentially
   reducing context to fit the model window.
4. Pass the resulting `messages` to completion or streaming completion methods.

`sources[].text` supports citation checks but does not guarantee the complete
context actually sent to the model. Disabling history reduces ambiguity without
removing the need to inspect the actual request.

The opt-in [step 10 adapter](../docs/step-10-context-capture.md) observes the
Ollama SDK `chat` request after message compression. It preserves original
messages and extracts document passages separately from instructions. Returned
sources remain citation evidence, not the source of evaluation contexts. This
is SDK-boundary observation for text-only requests, not a packet capture or
an observation of Ollama's internal tokenization.

[Reviewed source](https://github.com/Mintplex-Labs/anything-llm/blob/v1.16.2/server/utils/chats/stream.js)
