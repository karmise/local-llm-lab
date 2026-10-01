"""Combine independent checks with checksum-bound existing judge evidence."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

from llm_testkit import assertions
from llm_testkit.evaluation.faithfulness import load_sample


def check_faithfulness_evidence(evidence: dict[str, Any], checksum: str) -> dict[str, Any]:
    if evidence.get("schema_version") != 1 or evidence.get("metric") != "faithfulness":
        raise ValueError("Unsupported faithfulness report")
    if evidence.get("sample_sha256") != checksum:
        raise ValueError("Faithfulness evidence belongs to a different sample")
    if evidence.get("status") != "completed":
        raise ValueError("Faithfulness evaluation did not complete")
    result = evidence["result"]
    assertions.assert_quality_score(result["value"])
    statements, verdicts = result["statements"], result["verdicts"]
    if not statements or Counter(statements) != Counter(item["statement"] for item in verdicts):
        raise ValueError("Incomplete faithfulness claim evidence")
    if any(type(item["verdict"]) is not int or item["verdict"] not in (0, 1) for item in verdicts):
        raise ValueError("Invalid faithfulness verdict")
    recomputed = sum(item["verdict"] for item in verdicts) / len(verdicts)
    if not math.isclose(result["value"], recomputed, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Faithfulness score does not match its verdicts")
    return {
        "value": result["value"], "statements": statements, "verdicts": verdicts,
        "judge_model": evidence["judge_model"], "judge_model_digest": evidence["judge_model_digest"],
        "judge_configuration": evidence["judge_configuration"], "ragas_version": evidence["ragas_version"],
        "evaluated_at": evidence["created_at"], "threshold": None,
    }


def _dimension(name: str, check: Callable[[], Any], *, measured: bool = False) -> dict[str, Any]:
    dimension: dict[str, Any] = {"name": name}
    try:
        value = check()
        dimension["status"] = "measured" if measured else "passed"
        if value is not None:
            dimension["details"] = value
    except AssertionError as error:
        dimension.update(status="failed", error=str(error))
    except Exception as error:
        dimension.update(status="error", error=f"{type(error).__name__}: {error}")
    return dimension


def build_quality_report(sample_path: Path, evidence_path: Path, profile_path: Path) -> dict[str, Any]:
    sample, checksum = load_sample(sample_path)
    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    if profile.get("schema_version") != 1 or sample["user_input"] != profile["question"]:
        raise ValueError("Quality profile does not match the captured question")

    def source_check() -> dict[str, Any]:
        # Resolve the expected document from observed context, not response citations.
        titles = set()
        for context in sample["retrieved_contexts"]:
            metadata = re.match(r"<document_metadata>\n(.*?)\n</document_metadata>", context, re.DOTALL)
            if metadata:
                title = re.search(r"^sourceDocument: (.+)$", metadata[1], re.MULTILINE)
                if title and re.fullmatch(profile["document_title_pattern"], title[1]):
                    titles.add(title[1])
        if len(titles) != 1:
            raise ValueError("Expected exactly one policy document in captured context metadata")
        title = titles.pop()
        assertions.assert_document_sources(
            {"sources": sample.get("response_sources")}, document_title=title,
            fragments=profile["source_fragments"],
        )
        return {"expected_document_title": title}

    def faithfulness_check() -> dict[str, Any]:
        evidence = json.loads(evidence_path.read_text())
        return check_faithfulness_evidence(evidence, checksum)

    dimensions = [
        _dimension("Required answer facts", lambda: assertions.assert_required_facts(
            sample["response"], fact_patterns=profile["fact_patterns"],
        )),
        _dimension("Document sources", source_check),
        _dimension("Faithfulness measurement (no quality threshold)", faithfulness_check, measured=True),
    ]
    statuses = [dimension["status"] for dimension in dimensions]
    return {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": profile["id"], "sample_sha256": checksum,
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "faithfulness_report_path": str(evidence_path.resolve()),
        "status": "error" if "error" in statuses else "failed" if "failed" in statuses else "checks_passed",
        "interpretation": "Required facts and source checks; faithfulness is a recorded measurement, not a quality gate",
        "generation_model": sample["observation"]["request"]["model"],
        "question": sample["user_input"], "answer": sample["response"],
        "contexts": sample["retrieved_contexts"], "sources": sample.get("response_sources", []),
        "metadata": sample.get("metadata", {}), "dimensions": dimensions,
    }
