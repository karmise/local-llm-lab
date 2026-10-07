"""Scenario data builders and deterministic test doubles."""


def _case() -> dict:
    return {
        "id": "mixed",
        "response": "Leave is 23 days. Gym reimbursement is 5000.",
        "expected_score": 0.5,
        "claims": [{"pattern": "23", "verdict": 1}, {"pattern": "5000", "verdict": 0}],
    }


def _result() -> dict:
    return {
        "value": 0.5,
        "verdicts": [
            {"statement": "Leave is 23 days.", "verdict": 1},
            {"statement": "Gym reimbursement is 5000.", "verdict": 0},
        ],
    }
