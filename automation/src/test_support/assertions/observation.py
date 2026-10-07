"""Domain expectations for observation unit scenarios."""


def check_capture_hook_results(actual):
    for field in ("sameRequest", "sameReturn", "sameStream", "sameError"):
        assert actual[field] is True
