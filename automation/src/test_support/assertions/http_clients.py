"""Domain expectations for http clients unit scenarios."""

import pytest
import requests


def check_upload_closes_document_even_when_request_fails_outcome(api, fail, path):
    if fail:
        with pytest.raises(requests.Timeout):
            api.upload_document(path, "folder", filename="unique-policy.txt", timeout=42)
    else:
        api.upload_document(path, "folder", filename="unique-policy.txt", timeout=42)


def check_upload_arguments(filename, document, content_type, timeout):
    assert filename == "unique-policy.txt"
    assert document.read() == b"Fictional policy"
    assert content_type == "text/plain"
    assert timeout == 42
