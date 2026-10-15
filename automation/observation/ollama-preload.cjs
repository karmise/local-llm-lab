// Observe the installed Ollama SDK boundary. The test's capture marker is removed before the call, so the model
// receives the prompt AnythingLLM builds without the test and identical runs send identical input; nothing else in
// the request changes, and the capture records the request exactly as forwarded.
const Module = require("node:module");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const originalLoad = Module._load;
const patched = Symbol("llm-testkit-observation");
const marker = /\n?\[LLM_TESTKIT_CAPTURE:([a-f0-9]{32})\]/g;

const isSystem = (message) => message.role === "system" && typeof message.content === "string";

Module._load = function (...args) {
  const exported = originalLoad.apply(this, args);
  if (args[0] !== "ollama" || !exported.Ollama || exported.Ollama[patched]) return exported;
  const originalChat = exported.Ollama.prototype.chat;
  exported.Ollama.prototype.chat = function (request, ...rest) {
    const markers = (request.messages || []).filter(isSystem).flatMap((message) => [...message.content.matchAll(marker)]);
    if (markers.length !== 1) return originalChat.call(this, request, ...rest);
    const directory = process.env.LLM_TESTKIT_CAPTURE_DIR;
    if (!directory) throw new Error("Test capture directory is not configured");
    const captureId = markers[0][1];
    const forwarded = {
      ...request,
      messages: request.messages.map((message) => isSystem(message)
        ? { ...message, content: message.content.replace(marker, "") } : message),
    };
    fs.writeFileSync(path.join(directory, `${captureId}-${crypto.randomUUID()}.json`), JSON.stringify({
      schema_version: 2,
      boundary: "ollama-sdk-chat",
      captured_at: new Date().toISOString(),
      capture_id: captureId,
      request: forwarded,
    }, null, 2), {
      flag: "wx",
      // Disposable CI exports fictional context to a host with a different UID.
      mode: process.env.GITHUB_ACTIONS === "true" &&
        process.env.LLM_TESTKIT_CAPTURE_SHARED_READ === "true" ? 0o644 : 0o600,
    });
    return originalChat.call(this, forwarded, ...rest);
  };
  exported.Ollama[patched] = true;
  return exported;
};
