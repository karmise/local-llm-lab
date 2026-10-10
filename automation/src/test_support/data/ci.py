"""CI scope and negative provenance cases, kept outside test scenarios."""

REVISION = "a" * 40
OTHER_REVISION = "b" * 40
PRESETS = (("smoke", "primary", 2, 19), ("curated", "primary", 4, 43), ("smoke", "comparison", 4, 32),
        ("curated", "comparison", 8, 80))
INVALID_IDENTITY = ("revision", "source", "profile", "baseline", "plan", "generation_weights", "judge_weights", "test")
MODEL_NAMES = ("qwen3.5:4b", "bge-m3:567m")
REVIEWED_MODELS = ("qwen3.5:4b", "qwen2.5:7b", "bge-m3:567m")
BLOB_BYTES = b"reviewed fake model bytes"
INVALID_BLOBS = (b"wrong", b"x" * len(BLOB_BYTES), BLOB_BYTES + b"extra")
CAPTURE_PERMISSIONS = (("true", "true", 0o644), ("false", "true", 0o600), ("true", "false", 0o600))
CAPTURE_SCRIPT = r"""
const Module = require('node:module');
const fs = require('node:fs');
const load = Module._load;
class Ollama { chat() { return 'original-result'; } }
Module._load = function(name, ...rest) { return name === 'ollama' ? { Ollama } : load.call(this, name, ...rest); };
process.env.LLM_TESTKIT_CAPTURE_DIR = process.argv[2];
require(process.argv[1]);
const Client = require('ollama').Ollama;
const result = new Client().chat({messages: [{role: 'system', content: '[LLM_TESTKIT_CAPTURE:' + 'a'.repeat(32) + ']'}]});
const saved = fs.readdirSync(process.argv[2])[0];
console.log(JSON.stringify({mode: fs.statSync(process.argv[2] + '/' + saved).mode & 511, result}));
"""

REPORT_SECRET = "DISPOSABLE-TEST-CREDENTIAL"
REPORT_TRACE = f"Authorization: Bearer {REPORT_SECRET}\nReadTimeout: request failed\n{REPORT_SECRET}\n"
REPORT_XML = f'<testsuite><testcase name="test_golden_policy_answer[paid_leave-qwen3.5:4b]"><failure message="ReadTimeout">{REPORT_SECRET}</failure></testcase></testsuite>'
