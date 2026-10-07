"""Domain expectations for qualification unit scenarios."""


def check_trace_outcomes_outcome(status, trace):
    if status not in ("passed", "missing"):
        assert trace["deviations"][0]["details"][0]["text"] == "original detail"
