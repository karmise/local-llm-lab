import copy
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.steps import title
from test_support.builders.observation import (
    _capture,
    _sample,
    check_capture_hook_results,
    make_fail_stub,
    make_publish_stub,
    prepare_sample_rejects_mismatched_or_ambiguous_observations_case,
    prepare_sdk_hook_preserves_request_return_values_streams_and_errors_case,
)
from test_support.data.observation import (
    INVALID_EVIDENCE_NEVER_LEAVES_PARTIAL_OUTPUT_VALUE_CASES,
    SAMPLE_REJECTS_MISMATCHED_OR_AMBIGUOUS_OBSERVATIONS_CHANGE_CASES,
)
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit


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


@pytest.mark.parametrize("change", SAMPLE_REJECTS_MISMATCHED_OR_AMBIGUOUS_OBSERVATIONS_CHANGE_CASES)
@title("Captured sample rejects mismatched or ambiguous observations [{param_id}]")
def test_sample_rejects_mismatched_or_ambiguous_observations(change: str) -> None:
    capture = _capture()
    request = capture["request"]
    system = request["messages"][0]
    prepare_sample_rejects_mismatched_or_ambiguous_observations_case(change, request, system)
    with pytest.raises(ValueError):
        _sample(capture)


@title("SDK capture hook preserves requests, return values, streams and exceptions")
def test_sdk_hook_preserves_request_return_values_streams_and_errors(tmp_path: Path) -> None:
    node = shutil.which("node")
    prepare_sdk_hook_preserves_request_return_values_streams_and_errors_case(node)
    hook = AUTOMATION_ROOT / "src/llm_testkit/observation/ollama-preload.cjs"
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
    check_capture_hook_results(actual)
    assert actual["files"] == 1
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert saved["request"] == _capture()["request"]


@pytest.mark.parametrize("value", INVALID_EVIDENCE_NEVER_LEAVES_PARTIAL_OUTPUT_VALUE_CASES)
def test_invalid_evidence_never_leaves_partial_output(tmp_path, value):
    path = tmp_path / "sample.json"
    with pytest.raises((ValueError, TypeError)):
        write_sample(path, {"value": value})
    assert not list(tmp_path.iterdir())


def test_concurrent_evidence_writers_publish_once_without_overwrite(tmp_path):

    path = tmp_path / "sample.json"

    publish = make_publish_stub(path)

    with ThreadPoolExecutor(max_workers=4) as workers:
        outcomes = list(workers.map(publish, range(4)))
    winners = [i for i in outcomes if i is not None]
    assert len(winners) == 1
    assert json.loads(path.read_text())["writer"] == winners[0]
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [path]


def test_publication_failure_removes_temporary_evidence(tmp_path, monkeypatch):
    fail = make_fail_stub()

    monkeypatch.setattr("llm_testkit.observation.evaluation_sample.os.link", fail)
    with pytest.raises(OSError, match="publication unavailable"):
        write_sample(tmp_path / "sample.json", {"value": 1})
    assert not list(tmp_path.iterdir())
