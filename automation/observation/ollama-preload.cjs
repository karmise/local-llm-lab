// Observe the installed Ollama SDK boundary without changing request parameters.
const Module = require("node:module");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");
const originalLoad = Module._load;
const patched = Symbol("llm-testkit-observation");

Module._load = function (...args) {
  const exported = originalLoad.apply(this, args);
  if (args[0] !== "ollama" || !exported.Ollama || exported.Ollama[patched]) return exported;
  const originalChat = exported.Ollama.prototype.chat;
  exported.Ollama.prototype.chat = function (request, ...rest) {
    const markers = (request.messages || [])
      .filter((message) => message.role === "system" && typeof message.content === "string")
      .flatMap((message) => [...message.content.matchAll(/\[LLM_TESTKIT_CAPTURE:([a-f0-9]{32})\]/g)]);
    if (markers.length === 1) {
      const directory = process.env.LLM_TESTKIT_CAPTURE_DIR;
      if (!directory) throw new Error("Test capture directory is not configured");
      const filename = `${markers[0][1]}-${crypto.randomUUID()}.json`;
      fs.writeFileSync(path.join(directory, filename), JSON.stringify({
        schema_version: 1,
        boundary: "ollama-sdk-chat",
        captured_at: new Date().toISOString(),
        request,
      }, null, 2), {
        flag: "wx",
        // Disposable CI exports fictional context to a host with a different UID.
        mode: process.env.GITHUB_ACTIONS === "true" &&
          process.env.LLM_TESTKIT_CAPTURE_SHARED_READ === "true" ? 0o644 : 0o600,
      });
    }
    return originalChat.call(this, request, ...rest);
  };
  exported.Ollama[patched] = true;
  return exported;
};
