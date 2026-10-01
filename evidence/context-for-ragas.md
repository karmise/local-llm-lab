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

Before adding RAGAS, introduce a local observation point between AnythingLLM and
Ollama, or instrument the point after message compression in a separate setup.
A candidate experiment is an HTTP proxy capturing `/api/chat` request bodies.
Verify that it preserves request bodies, streaming and errors, and distinguish
document passages from instructions and history. The proxy is not installed and
full context capture has not yet been verified.

[Reviewed source](https://github.com/Mintplex-Labs/anything-llm/blob/v1.16.2/server/utils/chats/stream.js)
