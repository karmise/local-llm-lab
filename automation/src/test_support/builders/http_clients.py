"""Named scenario preparation and test doubles."""

import requests

from test_support.assertions.http_clients import check_upload_arguments


def make_upload_stub(documents, fail):
    def upload(**kwargs):
        filename, document, content_type = kwargs["files"]["file"]
        documents.append(document)
        check_upload_arguments(filename, document, content_type, kwargs["timeout"])
        if fail:
            raise requests.Timeout("Upload timed out")
        return requests.Response()

    return upload
