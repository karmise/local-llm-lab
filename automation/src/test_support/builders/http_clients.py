"""Named scenario preparation and test doubles."""

import pytest
import requests


def make_upload_stub(documents, fail):
    def upload(**kwargs):
        filename, document, content_type = kwargs["files"]["file"]
        documents.append(document)
        assert filename == "unique-policy.txt"
        assert document.read() == b"Fictional policy"
        assert content_type == "text/plain"
        assert kwargs["timeout"] == 42
        if fail:
            raise requests.Timeout("Upload timed out")
        return requests.Response()

    return upload


def check_upload_closes_document_even_when_request_fails_outcome(api, fail, path):
    if fail:
        with pytest.raises(requests.Timeout):
            api.upload_document(path, "folder", filename="unique-policy.txt", timeout=42)
    else:
        api.upload_document(path, "folder", filename="unique-policy.txt", timeout=42)
