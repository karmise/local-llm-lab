import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from llm_testkit.observation.evaluation_sample import build_sample, write_sample
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit
CAPTURE_ID = "a" * 32


def _capture() -> dict:
    return {
        "schema_version": 1,
        "boundary": "ollama-sdk-chat",
        "request": {
            "model": "qwen3.5:4b",
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Instructions\n[LLM_TESTKIT_CAPTURE:{CAPTURE_ID}]\nContext:\n"
                        "[CONTEXT 0]:\n23 working days\n[END CONTEXT 0]\n\n"
                        "[CONTEXT 1]:\n12 calendar days\n[END CONTEXT 1]\n\n"
                    ),
                },
                {"role": "user", "content": "Leave?"},
            ],
        },
    }


def _sample(capture: dict) -> dict:
    return build_sample(
        capture,
        question="Leave?",
        answer="23 working days",
        reference="Expected answer",
        expected_model="qwen3.5:4b",
        capture_id=CAPTURE_ID,
    )


@title("Captured sample extracts document context and preserves the original request")
def test_sample_extracts_only_actual_document_context_and_preserves_request(tmp_path: Path) -> None:
    capture = _capture()
    original = copy.deepcopy(capture)
    sample = _sample(capture)
    assert sample["retrieved_contexts"] == ["23 working days", "12 calendar days"]
    assert sample["observation"] == original
    path = tmp_path / "sample.json"
    write_sample(path, sample)
    assert json.loads(path.read_text())["reference"] == "Expected answer"
    with pytest.raises(FileExistsError):
        write_sample(path, sample)


@pytest.mark.parametrize(
    "change", ["model", "question", "history", "marker", "truncated", "indices", "empty"]
)
@title("Captured sample rejects mismatched or ambiguous observations [{param_id}]")
def test_sample_rejects_mismatched_or_ambiguous_observations(change: str) -> None:
    capture = _capture()
    request = capture["request"]
    system = request["messages"][0]
    if change == "model":
        request["model"] = "other-model"
    elif change == "question":
        request["messages"][1]["content"] = "Other question"
    elif change == "history":
        request["messages"].append({"role": "assistant", "content": "History"})
    elif change == "marker":
        system["content"] = system["content"].replace(CAPTURE_ID, "b" * 32)
    elif change == "truncated":
        system["content"] = system["content"].replace("[END CONTEXT 1]", "")
    elif change == "indices":
        system["content"] = system["content"].replace("CONTEXT 1", "CONTEXT 2")
    else:
        system["content"] = system["content"].replace("23 working days", "")
    with pytest.raises(ValueError):
        _sample(capture)


@title("SDK capture hook preserves requests, return values, streams and exceptions")
def test_sdk_hook_preserves_request_return_values_streams_and_errors(tmp_path: Path) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to verify the optional application preload")
    hook = Path(__file__).resolve().parents[2] / "src/llm_testkit/observation/ollama-preload.cjs"
    script = r"""
const Module = require("node:module");
const fs = require("node:fs");
const load = Module._load;
const response = Promise.resolve("response");
const stream = { [Symbol.asyncIterator]: async function* () { yield "chunk"; } };
const error = new Error("upstream failure");
let observed;
class Ollama {
  chat(request) { observed = request; if (request.fail) throw error; return request.stream ? stream : response; }
}
Module._load = function (name, ...rest) { return name === "ollama" ? { Ollama } : load.call(this, name, ...rest); };
process.env.LLM_TESTKIT_CAPTURE_DIR = process.argv[2];
require(process.argv[1]);
require("ollama"); require("ollama");
const client = new Ollama();
const request = JSON.parse(process.argv[3]);
const before = JSON.stringify(request);
const returned = client.chat(request);
const sameRequest = observed === request && JSON.stringify(request) === before;
const unmarked = { messages: [], stream: true };
const sameStream = client.chat(unmarked) === stream;
let sameError = false;
try { client.chat({ messages: [], fail: true }); } catch (caught) { sameError = caught === error; }
console.log(JSON.stringify({ sameRequest, sameReturn: returned === response, sameStream, sameError,
  files: fs.readdirSync(process.argv[2]).length }));
"""
    result = subprocess.run(
        [node, "-e", script, str(hook), str(tmp_path), json.dumps(_capture()["request"])],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = json.loads(result.stdout)
    for field in ("sameRequest", "sameReturn", "sameStream", "sameError"):
        assert actual[field] is True
    assert actual["files"] == 1
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["request"] == _capture()["request"]
