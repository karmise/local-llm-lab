import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from llm_testkit.observation.evaluation_sample import write_sample
from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import values as value_checks
from test_support.assertions.observation import check_capture_hook_results
from test_support.builders.observation import (
    _capture,
    _sample,
    make_fail_stub,
    make_publish_stub,
    make_unserializable_capture_payload,
    prepare_sample_rejects_mismatched_or_ambiguous_observations_case,
    prepare_sdk_hook_preserves_request_return_values_streams_and_errors_case,
)
from test_support.data import common as case_data
from test_support.data.observation import (
    INVALID_EVIDENCE_NEVER_LEAVES_PARTIAL_OUTPUT_VALUE_CASES,
    SAMPLE_REJECTS_MISMATCHED_OR_AMBIGUOUS_OBSERVATIONS_CHANGE_CASES,
    SERIALIZABLE_CAPTURE_PAYLOAD,
)
from test_support.paths import AUTOMATION_ROOT

pytestmark = pytest.mark.unit


@title("Captured sample extracts document context and preserves the original request")
def test_sample_extracts_only_actual_document_context_and_preserves_request(tmp_path: Path) -> None:
    capture = _capture()
    original = copy.deepcopy(capture)
    sample = _sample(capture)
    value_checks.equal(sample["retrieved_contexts"], ["23 working days", "12 calendar days"])
    value_checks.equal(sample["observation"], original)
    path = tmp_path / case_data.SAMPLE_FILE_NAME
    write_sample(path, sample)
    value_checks.equal(json.loads(path.read_text())["reference"], "Expected answer")
    errors.rejects(lambda: write_sample(path, sample), expected=FileExistsError)


@pytest.mark.parametrize("change", SAMPLE_REJECTS_MISMATCHED_OR_AMBIGUOUS_OBSERVATIONS_CHANGE_CASES)
@title("Captured sample rejects mismatched or ambiguous observations [{param_id}]")
def test_sample_rejects_mismatched_or_ambiguous_observations(change: str) -> None:
    capture = _capture()
    request = capture["request"]
    system = request["messages"][0]
    prepare_sample_rejects_mismatched_or_ambiguous_observations_case(change, request, system)
    errors.rejects(lambda: _sample(capture), expected=ValueError)


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
    value_checks.equal(actual["files"], 1)
    saved = json.loads(next(tmp_path.glob("*.json")).read_text())
    value_checks.equal(saved["request"], _capture()["request"])


@pytest.mark.parametrize("value", INVALID_EVIDENCE_NEVER_LEAVES_PARTIAL_OUTPUT_VALUE_CASES)
def test_invalid_evidence_never_leaves_partial_output(tmp_path, value):
    path = tmp_path / case_data.SAMPLE_FILE_NAME
    errors.rejects(
        lambda: write_sample(path, make_unserializable_capture_payload(value)),
        expected=(ValueError, TypeError),
    )
    value_checks.falsy(list(tmp_path.iterdir()))


def test_concurrent_evidence_writers_publish_once_without_overwrite(tmp_path, evidence_workers):

    path = tmp_path / case_data.SAMPLE_FILE_NAME

    publish = make_publish_stub(path)

    outcomes = list(evidence_workers.map(publish, range(4)))
    winners = [i for i in outcomes if i is not None]
    value_checks.length(winners, 1)
    value_checks.equal(json.loads(path.read_text())["writer"], winners[0])
    value_checks.equal(path.stat().st_mode & 511, 384)
    value_checks.equal(list(tmp_path.iterdir()), [path])


def test_publication_failure_removes_temporary_evidence(tmp_path, monkeypatch):
    fail = make_fail_stub()

    monkeypatch.setattr("llm_testkit.observation.evaluation_sample.os.link", fail)
    errors.rejects(
        lambda: write_sample(
            tmp_path / case_data.SAMPLE_FILE_NAME, case_data.fresh(SERIALIZABLE_CAPTURE_PAYLOAD)
        ),
        expected=OSError,
        match="publication unavailable",
    )
    value_checks.falsy(list(tmp_path.iterdir()))
