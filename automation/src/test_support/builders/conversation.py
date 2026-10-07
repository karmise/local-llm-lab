"""Scenario data builders and deterministic test doubles."""

import json

from requests import Response

from test_support.data.conversation import DATA as DATA


def response(answer: str, *, title: str = "policy.txt", source: str | None = None) -> Response:
    result = Response()
    result.status_code = 200
    result._content = json.dumps(
        {
            "type": "textResponse",
            "error": None,
            "close": True,
            "textResponse": answer,
            "sources": [
                {"title": title, "text": source or (DATA / "company-policy.txt").read_text()}
            ],
        }
    ).encode()
    return result


def prepare_invalid_catalogs_fail_before_any_generation_case(data, defect, row):
    if defect == "checksum":
        data["policy_sha256"] = "0" * 64
    elif defect == "duplicate id":
        data["cases"][1]["id"] = row["id"]
    elif defect == "word limit type":
        row["max_words"] = True
    elif defect == "absent source":
        row["source_fragments"] = ["Not in the policy"]
    elif defect == "missing reference rule":
        row["reference"] = "Ready to help."
    else:
        row["reference"] = "Hi! You have 23 working days of paid leave."


def make_defective_policy_source(case, defect):
    reply = response(
        case.reference,
        title="other.txt" if defect == "wrong document" else "policy.txt",
        source="Unrelated passage" if defect == "unsupported passage" else None,
    )

    return reply


def make_walking_response(case, case_id, suggestion):
    answer = (
        suggestion
        if case_id == "small_talk"
        else case.reference.replace("A walk can be a nice way to relax.", suggestion)
    )

    return answer
