"""Named scenario preparation and test doubles."""

import requests

from test_support.assertions.http_clients import check_upload_arguments
from test_support.data import common as case_data


def make_upload_stub(documents, fail):
    def upload(**kwargs):
        filename, document, content_type = kwargs["files"]["file"]
        documents.append(document)
        check_upload_arguments(filename, document, content_type, kwargs["timeout"])
        if fail:
            raise requests.Timeout("Upload timed out")
        return requests.Response()

    return upload


def make_workspace_template():
    """Build input for test_workspace_creation_does_not_mutate_configuration."""
    return {"name": "template", "chatModel": case_data.TEST_MODEL}


def make_created_workspace_payload():
    """Build input for test_workspace_creation_does_not_mutate_configuration."""
    return {"name": "temporary", "chatModel": case_data.TEST_MODEL}
