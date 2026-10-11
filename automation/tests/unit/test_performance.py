"""Performance: bounded closed-loop batches, validated evidence, recorded experiments and comparable baselines."""

import json
import sys
import threading
from copy import deepcopy
from datetime import datetime
from unittest.mock import Mock

import pytest

from llm_testkit import assertions
from llm_testkit.performance import comparison, reporting, runner
from llm_testkit.performance.comparison import compare_batches
from llm_testkit.performance.reporting import record_batch
from llm_testkit.performance.runner import run_batch, validate_batch, validate_budget
from llm_testkit.reporting.steps import title

pytestmark = pytest.mark.unit


@pytest.fixture
def clock(monkeypatch):
    """A perf_counter returning the given times in order; one user makes the order deterministic."""
    def set_times(*times):
        monkeypatch.setattr(runner.time, "perf_counter", Mock(side_effect=list(times)))

    return set_times


def timed_batch(clock, operation=lambda: None):
    """Four sequential attempts lasting 0.1, 0.4, 0.2 and 0.3 seconds within a one-second batch."""
    clock(0.0, 0.0, 0.1, 0.1, 0.5, 0.5, 0.7, 0.7, 1.0, 1.0)
    return run_batch(operation, requests=4)


@title("A batch records each attempt and summarises latency, throughput and failures")
def test_run_batch_summary(clock):
    report = timed_batch(clock)

    assert {
            k: v
            for k, v in report.items() if k not in ("created_at", "attempts", "latency_seconds")} == {
            "schema_version": 1,
            "kind": "performance_batch",
            "requests": 4,
            "users": 1,
            "failed": 0,
            "wall_seconds": 1.0,
            "completed_requests_per_second": 4.0,
            "interpretation": report["interpretation"]}
    assert [(row["index"], row["status"]) for row in report["attempts"]] == [(i, "passed") for i in range(4)]
    assert [row["elapsed_seconds"] for row in report["attempts"]] == pytest.approx([0.1, 0.4, 0.2, 0.3])
    assert report["latency_seconds"] == pytest.approx({"minimum": 0.1, "median": 0.25, "p95": 0.4, "maximum": 0.4})
    assert "not TTFT" in report["interpretation"]
    assert datetime.fromisoformat(report["created_at"]).utcoffset().total_seconds() == 0
    validate_batch(report)


@title("A failed attempt is retained with its error and is not retried")
def test_run_batch_retains_failures(clock):
    calls = []

    def operation():
        calls.append(1)
        if len(calls) == 2:
            raise TimeoutError("Model timed out")

    report = timed_batch(clock, operation)

    assert len(calls) == 4
    assert report["attempts"][1] == {
            "index": 1,
            "status": "failed",
            "error_type": "TimeoutError",
            "error": "Model timed out",
            "elapsed_seconds": pytest.approx(0.4)}
    assert (report["failed"], report["completed_requests_per_second"]) == (1, 3.0)


@title("Concurrent users execute exactly the request budget")
def test_run_batch_concurrent_users():
    lock, calls = threading.Lock(), []

    def operation():
        with lock:
            calls.append(threading.get_ident())

    report = run_batch(operation, requests=8, users=4)

    assert (len(calls), report["users"], [row["index"] for row in report["attempts"]]) == (8, 4, list(range(8)))
    validate_batch(report)


@pytest.mark.parametrize(("requests", "users", "message"), [
        pytest.param(0, 1, "request budget must be between one and twenty", id="no-requests"),
        pytest.param(21, 1, "request budget must be between one and twenty", id="too-many-requests"),
        pytest.param(True, 1, "request budget must be between one and twenty", id="boolean-requests"),
        pytest.param(1.0, 1, "request budget must be between one and twenty", id="float-requests"),
        pytest.param(2, 3, "Use one to four users, not exceeding request count", id="more-users-than-requests"),
        pytest.param(5, 5, "Use one to four users", id="five-users"),
        pytest.param(2, 0, "Use one to four users", id="no-users"),
        pytest.param(2, True, "Use one to four users", id="boolean-users")])
@title("An invalid load budget fails before any request [{param_id}]")
def test_invalid_budget(requests, users, message):
    calls = []

    with pytest.raises(ValueError, match=message):
        run_batch(lambda: calls.append(1), requests=requests, users=users)
    assert calls == []


@pytest.mark.parametrize(("requests", "users"), [(1, 1), (20, 4), (4, 4)])
@title("Budgets at the limits are accepted [{requests} requests, {users} users]")
def test_valid_budget(requests, users):
    assert validate_budget(requests, users) is None


def edit(path, value=None, *, delete=False):
    """An edit of a nested report field."""
    def apply(report):
        target = report
        for key in path[:-1]:
            target = target[key]
        if delete:
            del target[path[-1]]
        else:
            target[path[-1]] = value

    return apply


@pytest.mark.parametrize(("change", "message"), [
        pytest.param(edit(("schema_version", ), 2), "Unsupported performance evidence", id="schema-two"),
        pytest.param(edit(("schema_version", ), True), "Unsupported performance evidence", id="schema-boolean"),
        pytest.param(edit(("kind", ), "batch"), "Unsupported performance evidence", id="kind"),
        pytest.param(edit(("users", ), 5), "Use one to four users", id="saved-users"),
        pytest.param(edit(("requests", ), True), "request budget", id="saved-requests"),
        pytest.param(
        lambda r: r["attempts"].pop(), "Incomplete or duplicate performance attempts", id="missing-attempt"),
        pytest.param(edit(("attempts", 0), None), "Incomplete or duplicate", id="attempt-not-object"),
        pytest.param(edit(("attempts", 0, "index"), False), "Incomplete or duplicate", id="boolean-index"),
        pytest.param(edit(("attempts", 1, "index"), 0), "Incomplete or duplicate", id="duplicate-index"),
        pytest.param(edit(("attempts", ), {}), "Incomplete or duplicate", id="attempts-not-list"),
        pytest.param(edit(("attempts", 0, "status"), delete=True), "Invalid performance outcome", id="no-status"),
        pytest.param(edit(("attempts", 0, "status"), "skipped"), "Invalid performance outcome", id="other-status"),
        pytest.param(edit(("attempts", 0, "elapsed_seconds"), "0.1"), "Invalid latency evidence", id="text-latency"),
        pytest.param(
        edit(("attempts", 0, "elapsed_seconds"), float("nan")), "Invalid latency evidence", id="nan-latency"),
        pytest.param(edit(("attempts", 0, "elapsed_seconds"), -0.1), "Invalid latency evidence", id="negative-latency"),
        pytest.param(edit(("wall_seconds", ), True), "Invalid wall time", id="boolean-wall"),
        pytest.param(edit(("wall_seconds", ), 0), "Invalid wall time", id="zero-wall"),
        pytest.param(edit(("wall_seconds", ), float("inf")), "Invalid wall time", id="infinite-wall"),
        pytest.param(edit(("wall_seconds", ), 0.3), "Request latency exceeds batch wall time", id="wall-too-short"),
        pytest.param(edit(("failed", ), 1), "Performance failure count mismatch", id="failed-count"),
        pytest.param(edit(("failed", ), False), "Performance failure count mismatch", id="boolean-failed"),
        pytest.param(edit(("latency_seconds", "p95"), 99), "Latency summary differs", id="p95"),
        pytest.param(edit(("latency_seconds", "median"), True), "Latency summary differs", id="boolean-median"),
        pytest.param(edit(("latency_seconds", ), []), "Latency summary differs", id="summary-not-object"),
        pytest.param(edit(("completed_requests_per_second", ), 0), "Throughput summary mismatch", id="throughput"),
        pytest.param(
        edit(("completed_requests_per_second", ), True), "Throughput summary mismatch", id="boolean-throughput")])
@title("Saved performance evidence is rejected when incomplete or inconsistent [{param_id}]")
def test_validate_batch_rejects(clock, change, message):
    report = timed_batch(clock)
    change(report)

    with pytest.raises(ValueError, match=message):
        validate_batch(report)


@title("Evidence that is not an object is rejected")
def test_validate_batch_rejects_non_object():
    with pytest.raises(ValueError, match="Unsupported performance evidence"):
        validate_batch([])


@title("A failed batch fails the default zero failure-rate gate but passes a declared allowance")
def test_performance_gate_failure_rate(clock):
    report = timed_batch(clock, Mock(side_effect=[None, TimeoutError("x"), None, None]))

    with pytest.raises(AssertionError, match="Performance failure rate exceeds threshold"):
        assertions.assert_performance_batch(report, maximum_p95=10)
    assertions.assert_performance_batch(report, maximum_p95=10, maximum_failure_rate=0.25)


@title("The latency gate accepts a p95 equal to the threshold and rejects a lower threshold")
def test_performance_gate_latency(clock):
    report = timed_batch(clock)

    assertions.assert_performance_batch(report, maximum_p95=report["latency_seconds"]["p95"])
    with pytest.raises(AssertionError, match="Performance p95 exceeds threshold"):
        assertions.assert_performance_batch(report, maximum_p95=0.39)


@pytest.mark.parametrize("maximum_p95", [0, float("nan"), True, "1"])
@title("The p95 threshold must be finite and positive [{maximum_p95}]")
def test_performance_gate_rejects_invalid_threshold(clock, maximum_p95):
    with pytest.raises(AssertionError, match="Invalid p95 threshold"):
        assertions.assert_performance_batch(timed_batch(clock), maximum_p95=maximum_p95)


@pytest.fixture
def platform(monkeypatch):
    for name, value in (("python_version", "3.12.9"), ("platform", "test-os"), ("node", "test-machine")):
        monkeypatch.setattr(reporting.platform, name, lambda value=value: value)


@title("A recorded batch keeps its attempts, the actual timeout and the machine identity")
def test_record_batch(clock, tmp_path, platform):
    report = timed_batch(clock)
    original = deepcopy(report)

    path = record_batch(
            report, tmp_path, base_url="http://localhost", timeout=3, maximum_p95=10, metadata={"workload": "health"})

    saved = json.loads(path.read_text())
    assert path.parent == tmp_path and len(path.stem) == 32
    assert saved["metadata"] == {
            "workload": "health",
            "python": "3.12.9",
            "system": "test-os",
            "machine": "test-machine",
            "base_url": "http://localhost",
            "timeout": 3,
            "warmup_requests": 0}
    assert saved["thresholds"] == {"maximum_p95": 10, "maximum_failure_rate": 0.0}
    assert {k: v for k, v in saved.items() if k not in ("metadata", "thresholds")} == json.loads(json.dumps(original))
    assert report == original


@title("The recorded evidence is attached to the report as JSON")
def test_record_batch_attaches_evidence(clock, tmp_path, platform, monkeypatch):
    attach = Mock()
    monkeypatch.setattr(reporting, "attach_file", attach)

    path = record_batch(
            timed_batch(clock), tmp_path, base_url="http://localhost", timeout=3, maximum_p95=10, metadata={})

    attach.assert_called_once_with(
            path, name="Performance attempts and thresholds", media_type="application/json", extension="json")


@title("A batch that fails its gate is saved before the gate rejects it")
def test_record_failed_batch(clock, tmp_path, platform):
    report = timed_batch(clock, Mock(side_effect=TimeoutError("retained")))

    with pytest.raises(AssertionError, match="failure rate"):
        record_batch(report, tmp_path, base_url="http://localhost", timeout=3, maximum_p95=10, metadata={})

    saved, = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert (saved["failed"], saved["attempts"][0]["error"]) == (4, "retained")


@title("Invalid evidence is not recorded")
def test_record_rejects_invalid_batch(clock, tmp_path):
    report = timed_batch(clock)
    report["failed"] = 1

    with pytest.raises(ValueError, match="failure count mismatch"):
        record_batch(report, tmp_path, base_url="http://localhost", timeout=3, maximum_p95=10, metadata={})
    assert list(tmp_path.iterdir()) == []


HEALTH = {
        "workload": "health",
        "system": "test-os",
        "machine": "test-machine",
        "python": "3.12",
        "base_url": "http://localhost",
        "warmup_requests": 0,
        "timeout": 5}
RAG = HEALTH | {
        "workload": "rag",
        "timeout": 60,
        "policy_sha256": "a" * 64,
        "golden_dataset_sha256": "b" * 64,
        "case_id": "carryover_limit",
        "model_digest": "c" * 64,
        "generation_model": "model",
        "configuration": {
        "chatModel": "model",
        "topN": 4}}


@pytest.fixture
def batches(clock):
    """A healthy baseline batch and an identical current batch, with the given metadata."""
    def make(metadata=HEALTH, current_metadata=None):
        baseline = timed_batch(clock) | {"metadata": deepcopy(metadata)}
        current = deepcopy(baseline)
        current["metadata"] = deepcopy(metadata if current_metadata is None else current_metadata)
        return baseline, current

    return make


def slower(report, factor):
    """The same batch with every latency multiplied by ``factor``."""
    for row in report["attempts"]:
        row["elapsed_seconds"] *= factor
    report["latency_seconds"] = {k: v * factor for k, v in report["latency_seconds"].items()}
    report["wall_seconds"] *= factor
    report["completed_requests_per_second"] /= factor
    return report


@pytest.mark.parametrize("metadata", [pytest.param(HEALTH, id="health"), pytest.param(RAG, id="rag")])
@title("Identical comparable batches pass and the report names both p95 values [{param_id}]")
def test_comparison_passes(batches, metadata):
    baseline, current = batches(metadata)

    report = compare_batches(baseline, current)

    assert report == {
            "schema_version": 1,
            "status": "passed",
            "maximum_growth": 0.2,
            "p95_growth": 0.0,
            "baseline_p95": baseline["latency_seconds"]["p95"],
            "current_p95": current["latency_seconds"]["p95"],
            "incomparable_fields": [],
            "model_changed": False,
            "interpretation": report["interpretation"]}
    assert "not a statistically significant capacity estimate" in report["interpretation"]


@pytest.mark.parametrize(("factor", "status"), [
        pytest.param(1.2, "passed", id="at-allowed-growth"),
        pytest.param(1.21, "regression", id="beyond-allowed-growth"),
        pytest.param(0.5, "passed", id="faster")])
@title("Latency growth up to the allowance passes and beyond it is a regression [{param_id}]")
def test_comparison_growth(batches, factor, status):
    baseline, current = batches()

    report = compare_batches(baseline, slower(current, factor))

    assert (report["status"], report["p95_growth"]) == (status, pytest.approx(factor - 1))


@title("A custom growth allowance is applied")
def test_comparison_custom_growth(batches):
    baseline, current = batches()

    assert compare_batches(baseline, slower(current, 1.5), maximum_growth=0.5)["status"] == "passed"


@pytest.mark.parametrize("maximum_growth", [-0.1, 1.1, float("nan"), True])
@title("The growth allowance must be a finite fraction between zero and one [{maximum_growth}]")
def test_comparison_rejects_invalid_growth(batches, maximum_growth):
    baseline, current = batches()

    with pytest.raises(ValueError, match="Latency growth must be a finite fraction between zero and one"):
        compare_batches(baseline, current, maximum_growth=maximum_growth)


@title("A failed current attempt is a regression even without slowdown")
def test_comparison_current_failure_is_regression(batches):
    baseline, current = batches()
    current["attempts"][0].update(status="failed", error_type="TimeoutError", error="x")
    current["failed"] = 1
    current["completed_requests_per_second"] = 3.0

    assert compare_batches(baseline, current)["status"] == "regression"


@title("A baseline with failures cannot be compared")
def test_comparison_requires_healthy_baseline(batches):
    baseline, current = batches()
    baseline["attempts"][0].update(status="failed", error_type="TimeoutError", error="x")
    baseline["failed"] = 1
    baseline["completed_requests_per_second"] = 3.0

    report = compare_batches(baseline, current)

    assert (report["status"], report["incomparable_fields"]) == ("incomparable", ["baseline_not_healthy"])


@title("A baseline with zero p95 cannot be compared and reports no growth")
def test_comparison_requires_positive_baseline_p95(clock):
    clock(0.0, 0.0, 0.0, 1.0)
    baseline = run_batch(lambda: None) | {"metadata": HEALTH}
    current = deepcopy(baseline)

    report = compare_batches(baseline, current)

    assert (report["status"], report["p95_growth"],
            report["incomparable_fields"]) == ("incomparable", None, ["baseline_not_healthy"])


@pytest.mark.parametrize(("field", "value"), [
        pytest.param("workload", "rag-other", id="workload"),
        pytest.param("system", "other-os", id="system"),
        pytest.param("machine", "other-machine", id="machine"),
        pytest.param("python", "3.13", id="python"),
        pytest.param("base_url", "http://other", id="base-url"),
        pytest.param("warmup_requests", 1, id="warmup"),
        pytest.param("timeout", 10, id="timeout")])
@title("Health batches are comparable only under identical execution conditions [{param_id}]")
def test_health_comparison_requires_same_conditions(batches, field, value):
    baseline, current = batches(HEALTH, HEALTH | {field: value})

    report = compare_batches(baseline, current)

    assert report["status"] == "incomparable"
    assert field in report["incomparable_fields"]


@pytest.mark.parametrize(("field", "value"), [
        pytest.param("machine", " ", id="blank-machine"),
        pytest.param("python", 3.12, id="numeric-python"),
        pytest.param("warmup_requests", False, id="boolean-warmup"),
        pytest.param("warmup_requests", -1, id="negative-warmup"),
        pytest.param("timeout", 0, id="zero-timeout"),
        pytest.param("timeout", float("inf"), id="infinite-timeout"),
        pytest.param("timeout", True, id="boolean-timeout")])
@title("Invalid execution metadata on both sides is incomparable even when equal [{param_id}]")
def test_comparison_rejects_invalid_metadata(batches, field, value):
    baseline, current = batches(HEALTH | {field: value})

    assert field in compare_batches(baseline, current)["incomparable_fields"]


@title("A sub-second timeout is valid execution metadata")
def test_comparison_accepts_short_timeout(batches):
    assert compare_batches(*batches(HEALTH | {"timeout": 0.5}))["status"] == "passed"


@title("An unsupported workload is incomparable")
def test_comparison_rejects_unsupported_workload(batches):
    baseline, current = batches(HEALTH | {"workload": "load"})

    assert compare_batches(baseline, current)["incomparable_fields"] == ["unsupported_workload"]


@title("Batches without metadata are incomparable")
def test_comparison_requires_metadata(batches):
    baseline, current = batches()
    del current["metadata"]
    baseline["metadata"] = "health"

    report = compare_batches(baseline, current)

    assert report["incomparable_fields"] == [
            "workload", "system", "machine", "python", "base_url", "unsupported_workload", "warmup_requests", "timeout"]


@title("Batches with a different request count are incomparable")
def test_comparison_requires_same_requests(clock):
    baseline = timed_batch(clock) | {"metadata": HEALTH}
    clock(0.0, 0.0, 0.1, 1.0)
    current = run_batch(lambda: None) | {"metadata": HEALTH}

    assert "requests" in compare_batches(baseline, current)["incomparable_fields"]


@title("Batches with a different number of users are incomparable")
def test_comparison_requires_same_users():
    baseline = run_batch(lambda: None, requests=4, users=1) | {"metadata": HEALTH}
    current = run_batch(lambda: None, requests=4, users=2) | {"metadata": HEALTH}

    assert "users" in compare_batches(baseline, current)["incomparable_fields"]


@pytest.mark.parametrize(
        "field", [
        "timeout", "policy_sha256", "golden_dataset_sha256", "case_id", "model_digest", "generation_model",
        "configuration"])
@title("RAG batches require explicit provenance on both sides [{field}]")
def test_rag_comparison_requires_provenance(batches, field):
    metadata = {k: v for k, v in RAG.items() if k != field}

    report = compare_batches(*batches(metadata))

    assert (report["status"], field in report["incomparable_fields"]) == ("incomparable", True)


@pytest.mark.parametrize(("field", "value"), [
        pytest.param("policy_sha256", "A" * 64, id="uppercase-policy"),
        pytest.param("golden_dataset_sha256", "b" * 63, id="short-dataset"),
        pytest.param("case_id", " ", id="blank-case"),
        pytest.param("configuration", {}, id="empty-configuration"),
        pytest.param("configuration", {"chatModel": "other"}, id="configuration-for-other-model"),
        pytest.param("configuration", {
        "chatModel": "model",
        "openAiPrompt": 1}, id="invalid-prompt")])
@title("Invalid RAG provenance is incomparable [{param_id}]")
def test_rag_comparison_rejects_invalid_provenance(batches, field, value):
    report = compare_batches(*batches(RAG | {field: value}))

    assert field in report["incomparable_fields"]


@pytest.mark.parametrize(("field", "value"), [
        pytest.param("policy_sha256", "d" * 64, id="policy"),
        pytest.param("golden_dataset_sha256", "d" * 64, id="dataset"),
        pytest.param("case_id", "paid_leave", id="case"),
        pytest.param("configuration", {
        "chatModel": "model",
        "topN": 8}, id="configuration")])
@title("RAG batches under a different policy, dataset, case or configuration are incomparable [{param_id}]")
def test_rag_comparison_requires_same_provenance(batches, field, value):
    report = compare_batches(*batches(RAG, RAG | {field: value}))

    assert report["incomparable_fields"] == [field]


@title("A changed generation model is compared and flagged; the capture marker is ignored")
def test_rag_comparison_flags_model_change(batches):
    current = RAG | {
            "generation_model": "new",
            "model_digest": "e" * 64,
            "configuration": {
            "chatModel": "new",
            "topN": 4,
            "openAiPrompt": f"P\n[LLM_TESTKIT_CAPTURE:{'a' * 32}]"}}
    baseline_metadata = RAG | {"configuration": RAG["configuration"] | {"openAiPrompt": "P"}}

    report = compare_batches(*batches(baseline_metadata, current))

    assert (report["status"], report["model_changed"]) == ("passed", True)


def run_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["performance-compare", *map(str, args)])
    return comparison.main()


@pytest.mark.parametrize(("factor", "code", "status"),
        [pytest.param(1.0, 0, "passed", id="passed"),
        pytest.param(2.0, 1, "regression", id="regression")])
@title("The compare command writes the report and exits non-zero unless the comparison passed [{param_id}]")
def test_cli_compare(batches, tmp_path, monkeypatch, capsys, factor, code, status):
    baseline, current = batches()
    (tmp_path / "base.json").write_text(json.dumps(baseline))
    (tmp_path / "current.json").write_text(json.dumps(slower(current, factor)))

    exit_code = run_cli(
            monkeypatch, tmp_path / "base.json", tmp_path / "current.json", "--output", tmp_path / "out.json")

    assert exit_code == code
    assert json.loads((tmp_path / "out.json").read_text())["status"] == status
    assert capsys.readouterr().out == f"Performance comparison: {status}\n"


@title("The compare command passes the growth allowance through")
def test_cli_compare_maximum_growth(batches, tmp_path, monkeypatch):
    baseline, current = batches()
    (tmp_path / "base.json").write_text(json.dumps(baseline))
    (tmp_path / "current.json").write_text(json.dumps(slower(current, 1.5)))

    exit_code = run_cli(
            monkeypatch, tmp_path / "base.json", tmp_path / "current.json", "--maximum-growth", "0.5", "--output",
            tmp_path / "out.json")

    assert (exit_code, json.loads((tmp_path / "out.json").read_text())["maximum_growth"]) == (0, 0.5)


@title("The compare command refuses to overwrite a report and requires an output")
def test_cli_compare_output_rules(tmp_path, monkeypatch, capsys):
    (tmp_path / "out.json").write_text("{}")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as existing:
        run_cli(monkeypatch, "a.json", "b.json", "--output", "out.json")
    with pytest.raises(SystemExit) as missing:
        run_cli(monkeypatch, "a.json", "b.json")

    assert (existing.value.code, missing.value.code) == (2, 2)
    assert "Output already exists" in capsys.readouterr().err
    assert (tmp_path / "out.json").read_text() == "{}"
