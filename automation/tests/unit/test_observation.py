"""Observation: evaluation samples built from the observed model request, and evidence written atomically once."""

import copy
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import pytest

from llm_testkit.observation import evaluation_sample
from llm_testkit.observation.evaluation_sample import build_sample, write_sample
from llm_testkit.reporting.steps import title
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit

CAPTURE_ID = "a" * 32
SYSTEM = (
        f"Instructions\n[LLM_TESTKIT_CAPTURE:{CAPTURE_ID}]\nContext:\n"
        "[CONTEXT 0]:\n23 working days\n[END CONTEXT 0]\n\n"
        "[CONTEXT 1]:\n12 calendar days\nsecond line\n[END CONTEXT 1]\n\n")


def capture(system: str = SYSTEM) -> dict:
    """An observed non-streaming chat request with one system and one user message."""
    return {
            "schema_version": 1,
            "boundary": "ollama-sdk-chat",
            "request": {
            "model": "qwen3.5:4b",
            "stream": False,
            "messages": [{
            "role": "system",
            "content": system}, {
            "role": "user",
            "content": "Leave?"}]}}


def sample(observed: dict) -> dict:
    return build_sample(
            observed, question="Leave?", answer="23 working days", reference="Expected answer",
            expected_model="qwen3.5:4b", capture_id=CAPTURE_ID)


@title("A sample takes its contexts from the observed request and keeps the request unchanged")
def test_sample_from_observed_request():
    observed = capture()
    original = copy.deepcopy(observed)

    result = sample(observed)

    assert result == {
            "schema_version": 1,
            "user_input": "Leave?",
            "retrieved_contexts": ["23 working days", "12 calendar days\nsecond line"],
            "response": "23 working days",
            "reference": "Expected answer",
            "observation": original,
            "capture_id": CAPTURE_ID,
            "context_parser": "anythingllm-1.16.2-ollama"}
    assert observed == original


def change(path, value):
    """Return an edit that sets the nested field at ``path`` of a capture."""
    def edit(observed):
        target = observed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return edit


def system(text):
    return change(("request", "messages", 0, "content"), text)


@pytest.mark.parametrize(("edit", "message"), [
        pytest.param(change(("schema_version", ), 2), "Unsupported observation boundary or schema", id="schema"),
        pytest.param(change(("boundary", ), "http"), "Unsupported observation boundary or schema", id="boundary"),
        pytest.param(
        change(("request", "model"), "other-model"), "selected model and non-streaming chat", id="other-model"),
        pytest.param(change(("request", "stream"), True), "selected model and non-streaming chat", id="streaming"),
        pytest.param(change(("request", "stream"), None), "selected model and non-streaming chat", id="stream-unset"),
        pytest.param(
        lambda c: c["request"]["messages"].append({
        "role": "assistant",
        "content": "History"}), "without history", id="history"),
        pytest.param(lambda c: c["request"]["messages"].pop(0), "without history", id="no-system"),
        pytest.param(change(("request", "messages", 0, "role"), "user"), "without history", id="two-user-messages"),
        pytest.param(
        change(("request", "messages", 1, "content"), "Other question"),
        "Observed question does not match the scenario", id="other-question"),
        pytest.param(
        system(SYSTEM.replace(CAPTURE_ID, "b" * 32)), "Missing or ambiguous test capture marker", id="other-marker"),
        pytest.param(
        system(SYSTEM + f"[LLM_TESTKIT_CAPTURE:{CAPTURE_ID}]"), "Missing or ambiguous test capture marker",
        id="two-markers"),
        pytest.param(
        system(SYSTEM.replace("[END CONTEXT 1]", "")), "truncated or unsupported document context", id="truncated"),
        pytest.param(
        system(SYSTEM.replace("[CONTEXT 0]:\n", "[CONTEXT 0]:")), "truncated or unsupported",
        id="no-newline-after-opening"),
        pytest.param(system(SYSTEM + "[END CONTEXT 3]"), "truncated or unsupported", id="stray-end"),
        pytest.param(
        system(f"[LLM_TESTKIT_CAPTURE:{CAPTURE_ID}] no contexts"), "truncated or unsupported", id="no-context"),
        pytest.param(
        system(SYSTEM.replace("CONTEXT 1", "CONTEXT 2")), "Context indices must be contiguous and ordered", id="gap"),
        pytest.param(system(SYSTEM.replace("CONTEXT 0", "CONTEXT 9")), "contiguous and ordered", id="not-from-zero"),
        pytest.param(system(SYSTEM.replace("23 working days", " ")), "Empty document context", id="blank-context")])
@title("A mismatched or ambiguous observation is rejected rule by rule [{param_id}]")
def test_sample_rejects_invalid_observation(edit, message):
    observed = capture()
    edit(observed)

    with pytest.raises(ValueError, match=message):
        sample(observed)


@title("A written sample is pretty-printed UTF-8 JSON ending in a newline")
def test_write_sample(tmp_path):
    path = tmp_path / "reports/nested/sample.json"

    write_sample(path, {"answer": "23 дня", "value": 1})

    assert path.read_text(encoding="utf-8") == '{\n  "answer": "23 дня",\n  "value": 1\n}\n'
    assert path.stat().st_mode & 0o777 == 0o600


@title("Evidence is never overwritten")
def test_write_sample_is_exclusive(tmp_path):
    path = tmp_path / "sample.json"
    write_sample(path, {"writer": 1})

    with pytest.raises(FileExistsError):
        write_sample(path, {"writer": 2})
    assert json.loads(path.read_text()) == {"writer": 1}
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(("value", "error"), [
        pytest.param(float("nan"), ValueError, id="nan"),
        pytest.param(float("inf"), ValueError, id="infinity"),
        pytest.param(object(), TypeError, id="unserializable")])
@title("Evidence that is not valid JSON leaves no output [{param_id}]")
def test_invalid_evidence_leaves_no_output(tmp_path, value, error):
    with pytest.raises(error):
        write_sample(tmp_path / "sample.json", {"value": value})
    assert list(tmp_path.iterdir()) == []


@title("Concurrent writers publish exactly once, completely and without overwriting")
def test_concurrent_writers_publish_once(tmp_path):
    path = tmp_path / "sample.json"

    def publish(index):
        try:
            write_sample(path, {"writer": index, "payload": "content" * 1000})
            return index
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=4) as workers:
        outcomes = list(workers.map(publish, range(4)))

    winner, = [index for index in outcomes if index is not None]
    assert json.loads(path.read_text()) == {"writer": winner, "payload": "content" * 1000}
    assert list(tmp_path.iterdir()) == [path]


@title("Evidence is written to a hidden temporary file beside the target, then linked into place")
def test_write_sample_publishes_from_same_directory(tmp_path, monkeypatch):
    links = []
    real_link = os.link

    def link(source, target):
        links.append((source, target, source.read_text()))
        real_link(source, target)

    monkeypatch.setattr(evaluation_sample.os, "link", link)

    write_sample(tmp_path / "sample.json", {"value": 1})

    (source, target, content), = links
    assert (source.parent, target) == (tmp_path, tmp_path / "sample.json")
    assert source.name.startswith(".evidence-") and source.name.endswith(".tmp")
    assert content == '{\n  "value": 1\n}\n'
    assert not source.exists()


@title("A failed publication removes the temporary file")
def test_failed_publication_removes_temporary_file(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("publication unavailable")

    monkeypatch.setattr(evaluation_sample.os, "link", fail)

    with pytest.raises(OSError, match="publication unavailable"):
        write_sample(tmp_path / "sample.json", {"value": 1})
    assert list(tmp_path.iterdir()) == []


HOOK_SCRIPT = r"""
const Module = require("node:module");
const fs = require("node:fs");
const load = Module._load;
const response = Promise.resolve("response");
const stream = { [Symbol.asyncIterator]: async function* () { yield "chunk"; } };
const error = new Error("upstream failure");
let observed;
let calls = 0;
class Ollama {
  chat(request, options) { calls += 1; observed = [request, options]; if (request.fail) throw error;
    return request.stream ? stream : response; }
}
Module._load = function (name, ...rest) { return name === "ollama" ? { Ollama } : load.call(this, name, ...rest); };
require(process.argv[1]);
require("ollama"); require("ollama");
const client = new Ollama();
const request = JSON.parse(process.argv[2]);
const before = JSON.stringify(request);
const returned = client.chat(request, "option");
const result = { sameRequest: observed[0] === request && JSON.stringify(request) === before,
  sameOptions: observed[1] === "option", sameReturn: returned === response, calls };
result.sameStream = client.chat({ messages: [], stream: true }) === stream;
try { client.chat({ messages: [], fail: true }); } catch (caught) { result.sameError = caught === error; }
const marker = "[LLM_TESTKIT_CAPTURE:" + "b".repeat(32) + "]";
client.chat({ messages: [{ role: "system", content: marker + marker }] });
client.chat({ messages: [{ role: "user", content: marker }] });
result.calls = calls;
result.files = process.env.LLM_TESTKIT_CAPTURE_DIR ? fs.readdirSync(process.env.LLM_TESTKIT_CAPTURE_DIR) : [];
console.log(JSON.stringify(result));
"""


def run_hook(*, directory=None):
    node = shutil.which("node")
    assert node, "Node.js is required to verify the application preload"
    variables = {k: v for k, v in os.environ.items() if k != "LLM_TESTKIT_CAPTURE_DIR"}
    variables |= {"LLM_TESTKIT_CAPTURE_DIR": str(directory)} if directory else {}
    hook = AUTOMATION_ROOT / "src/llm_testkit/observation/ollama-preload.cjs"
    return subprocess.run(
            [node, "-e", HOOK_SCRIPT, str(hook), json.dumps(capture()["request"])], env=variables, capture_output=True,
            text=True)


@title("The SDK hook saves one capture of a marked request and preserves requests, returns, streams and errors")
def test_sdk_hook(tmp_path):
    result = run_hook(directory=tmp_path)

    observed = json.loads(result.stdout)
    assert observed == {
            "sameRequest": True,
            "sameOptions": True,
            "sameReturn": True,
            "calls": 5,
            "sameStream": True,
            "sameError": True,
            "files": observed["files"]}
    file, = observed["files"]
    assert file.startswith(f"{CAPTURE_ID}-") and file.endswith(".json")
    saved = json.loads((tmp_path / file).read_text())
    assert (saved["schema_version"], saved["boundary"],
            saved["request"]) == (1, "ollama-sdk-chat", capture()["request"])
    assert saved["captured_at"].endswith("Z")


@title("A marked request without a capture directory fails instead of silently skipping evidence")
def test_sdk_hook_requires_capture_directory():
    result = run_hook()

    assert result.returncode != 0
    assert "Test capture directory is not configured" in result.stderr
