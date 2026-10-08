"""Domain expectations for reporting steps unit scenarios."""


def check_reported_operation_outcome(backend):
    if backend is not None:
        assert backend.attach.call_count == 0
        assert backend.dynamic.parameter.call_count == 0
