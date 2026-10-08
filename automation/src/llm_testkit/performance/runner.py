"""Closed-loop request batches with retained failures and explicit latency gates."""

import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from statistics import median
from typing import Any


def validate_budget(requests: Any, users: Any) -> None:
    if type(requests) is not int or not 1 <= requests <= 20:
        raise ValueError("Performance request budget must be between one and twenty")
    if type(users) is not int or not 1 <= users <= min(requests, 4):
        raise ValueError("Use one to four users, not exceeding request count")


def run_batch(operation: Callable[[], None], *, requests: int = 1, users: int = 1) -> dict[str, Any]:
    validate_budget(requests, users)

    def attempt(index: int) -> dict[str, Any]:
        start = time.perf_counter()
        row = {"index": index, "status": "passed"}
        try:
            operation()
        except Exception as error:
            row.update(status="failed", error_type=type(error).__name__, error=str(error))
        row["elapsed_seconds"] = time.perf_counter() - start
        return row

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=users) as executor:
        attempts = list(executor.map(attempt, range(requests)))
    elapsed = time.perf_counter() - started
    values = sorted(row["elapsed_seconds"] for row in attempts)
    completed = sum(row["status"] == "passed" for row in attempts)
    return {
            "schema_version":
            1,
            "kind":
            "performance_batch",
            "created_at":
            datetime.now(timezone.utc).isoformat(),
            "requests":
            requests,
            "users":
            users,
            "attempts":
            attempts,
            "failed":
            requests - completed,
            "wall_seconds":
            elapsed,
            "completed_requests_per_second":
            completed / elapsed,
            "latency_seconds": {
            "minimum": values[0],
            "median": median(values),
            "p95": values[math.ceil(0.95 * requests) - 1],
            "maximum": values[-1]},
            "interpretation":
            "Bounded closed-loop batch; end-to-end non-streaming HTTP latency including validation; not TTFT or server-only inference time"
    }


def validate_batch(report: dict[str, Any]) -> None:
    if (not isinstance(report, dict) or type(report.get("schema_version")) is not int or report["schema_version"] != 1
                or report.get("kind") != "performance_batch"):
        raise ValueError("Unsupported performance evidence")
    validate_budget(report.get("requests"), report.get("users"))
    rows = report.get("attempts")
    if (not isinstance(rows, list) or len(rows) != report["requests"]
                or any(not isinstance(r, dict) or type(r.get("index")) is not int for r in rows)
                or [r["index"] for r in rows] != list(range(report["requests"]))):
        raise ValueError("Incomplete or duplicate performance attempts")
    if any(r.get("status") not in ("passed", "failed") for r in rows):
        raise ValueError("Invalid performance outcome")
    values = [r.get("elapsed_seconds") for r in rows]
    if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Invalid latency evidence")
    values.sort()
    wall = report.get("wall_seconds")
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
        raise ValueError("Invalid wall time")
    if values[-1] > wall:
        raise ValueError("Request latency exceeds batch wall time")
    failed = sum(r["status"] == "failed" for r in rows)
    if type(report.get("failed")) is not int or report["failed"] != failed:
        raise ValueError("Performance failure count mismatch")
    expected = {
            "minimum": values[0],
            "median": median(values),
            "p95": values[math.ceil(0.95 * len(values)) - 1],
            "maximum": values[-1]}
    summary = report.get("latency_seconds")
    if (not isinstance(summary, dict) or any(type(v) not in (int, float) for v in summary.values())
                or summary != expected):
        raise ValueError("Latency summary differs from individual attempts")
    rps = (len(rows) - failed) / wall
    actual_rps = report.get("completed_requests_per_second")
    if type(actual_rps) not in (int, float) or not math.isclose(actual_rps, rps):
        raise ValueError("Throughput summary mismatch")
