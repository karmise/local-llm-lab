"""Named expectations for judge suitability and preserved failures."""

from test_support.assertions import values


def matched_control(results: list[dict]) -> None:
    values.length(results, 1)
    values.equal(results[0]["status"], "matched")
    values.equal(results[0]["mismatches"], [])


def failures_are_retained(results: list[dict]) -> None:
    values.equal([row["status"] for row in results], ["matched", "mismatch", "error", "matched"])
    values.equal(results[2]["error"]["type"], "ValueError")
    values.length(results[2]["judge_calls"], 1)


def experimental_decision(summary: dict) -> None:
    values.equal(summary["status"], "error")
    values.equal(summary["counts"], {"matched": 2, "mismatch": 1, "error": 1})
    values.equal(summary["release_suitability"], "not_established")
    values.equal(summary["human_review"], "pending")
    values.equal(summary["threshold_decision"], "retain_experimental_thresholds")


def unapproved_review(report: dict) -> None:
    values.equal(report["human_review"], "pending")
    values.equal(report["threshold_decision"], "retain_experimental_thresholds")
    values.equal(report["original_benchmark_status"], "failed")
    values.length(report["rows"], 2)
    values.all_true(row["human_verdict"] is None for row in report["rows"])
    values.equal(report["rows"][0]["assessment"], "judge_reference_disagreement_candidate")
    values.equal(report["rows"][0]["evidence"]["correctness"]["result"]["value"], 0.0)
    values.length(report["rows"][0]["evidence"]["correctness"]["sha256"], 64)
