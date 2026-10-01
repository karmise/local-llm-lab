"""Evaluate one captured sample; retain claim verdicts and failure evidence."""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

from llm_testkit import assertions
from llm_testkit.clients.ollama_client import OllamaClient
from llm_testkit.config import Settings
from llm_testkit.core.http_client import HttpClient
from llm_testkit.observation.evaluation_sample import build_sample, write_sample


def load_sample(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    sample = json.loads(raw)
    # Rebuild from the saved observation to reject edited or unrelated contexts.
    rebuilt = build_sample(
        sample["observation"], question=sample["user_input"], answer=sample["response"],
        reference=sample["reference"],
        expected_model=sample["observation"]["request"]["model"],
        capture_id=sample["capture_id"],
    )
    if sample.get("schema_version") != 1 or sample["retrieved_contexts"] != rebuilt["retrieved_contexts"]:
        raise ValueError("Evaluation contexts do not match the saved observation")
    for field in ("user_input", "response", "reference"):
        if not isinstance(sample[field], str) or not sample[field].strip():
            raise ValueError(f"Evaluation sample requires a nonempty {field}")
    return sample, hashlib.sha256(raw).hexdigest()


async def score_sample(sample: dict[str, Any], judge: Any) -> dict[str, Any]:
    from ragas.metrics.collections import Faithfulness

    result = await Faithfulness(llm=judge).ascore(
        user_input=sample["user_input"], response=sample["response"],
        retrieved_contexts=sample["retrieved_contexts"],
    )
    assertions.assert_quality_score(result.value)
    if len(judge.calls) != 2:
        raise ValueError("Expected statement extraction and claim verification")
    statements = judge.calls[0]["output"]["statements"]
    verdicts = judge.calls[1]["output"]["statements"]
    if not statements or Counter(statements) != Counter(item["statement"] for item in verdicts):
        raise ValueError("Judge verdicts must cover every extracted statement exactly once")
    if any(type(item["verdict"]) is not int or item["verdict"] not in (0, 1) for item in verdicts):
        raise ValueError("Judge verdict must be 0 or 1")
    return {"value": result.value, "statements": statements, "verdicts": verdicts}


def evaluate_sample_report(
    sample_path: Path, *, settings: Settings, judge_model: str = "qwen3.5:4b",
) -> dict[str, Any]:
    """Run at most two judge calls and preserve completed or failed evidence."""
    from llm_testkit.evaluation.ollama_judge import OllamaJudge

    report: dict[str, Any] = {
        "schema_version": 1, "metric": "faithfulness", "status": "error",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_path": str(sample_path.resolve()), "ragas_version": version("ragas"),
        "judge_model": judge_model, "threshold": None,
        "interpretation": "Exploratory judge result; not a calibrated quality gate",
    }
    judge = None
    http = None
    try:
        sample, checksum = load_sample(sample_path)
        report["sample_sha256"] = checksum
        report["generation_model"] = sample["observation"]["request"]["model"]
        report["same_generation_and_judge_model"] = report["generation_model"] == judge_model
        http = HttpClient(settings.ollama_base_url, settings.http_timeout)
        client = OllamaClient(http)
        catalog = client.list_models()
        assertions.assert_status_code(catalog, 200, context="Judge model catalog")
        report["judge_model_digest"] = assertions.assert_model_available(catalog.json()["models"], judge_model)
        judge = OllamaJudge(client, judge_model, settings.llm_timeout)
        report["judge_configuration"] = {"options": judge.options, "think": False, "retries": 0}
        report["result"] = asyncio.run(score_sample(sample, judge))
        report["status"] = "completed"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if judge:
            report["judge_calls"] = judge.calls
        if http:
            http.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path)
    parser.add_argument("--judge-model", default="qwen3.5:4b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new file")
    try:
        import_module("llm_testkit.evaluation.ollama_judge")
    except ImportError:
        parser.error('Install evaluation dependencies: python -m pip install -e ".[evaluation]"')
    report = evaluate_sample_report(args.sample, settings=Settings.from_env(), judge_model=args.judge_model)
    write_sample(args.output, report)
    if report["status"] != "completed":
        print(f"Evaluation failed; details saved to {args.output}")
        return 1
    print(f"Faithfulness: {report['result']['value']:.3f}; details saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
