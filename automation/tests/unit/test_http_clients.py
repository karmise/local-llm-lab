from pathlib import Path
from unittest.mock import Mock
from urllib.parse import quote

import pytest
import requests

from llm_testkit.reporting.steps import title
from test_support.assertions import errors as errors
from test_support.assertions import mocks as mock_checks
from test_support.assertions import values as value_checks
from test_support.assertions.http_clients import (
    check_upload_closes_document_even_when_request_fails_outcome,
)
from test_support.builders.http_clients import (
    make_created_workspace_payload,
    make_upload_stub,
    make_workspace_template,
)
from test_support.data import common as case_data
from test_support.data.http_clients import (
    AUTHENTICATED_HEADERS,
    CHAT_MESSAGE_PAYLOAD,
    REVIEWED_CHAT_CONFIGURATION,
    UPLOAD_CLOSES_DOCUMENT_EVEN_WHEN_REQUEST_FAILS_FAIL_CASES,
    WORKSPACE_SLUGS_ARE_ENCODED_AS_ONE_PATH_SEGMENT_SLUG_CASES,
)
from test_support.fixtures.unit_http_clients import session as session

pytestmark = pytest.mark.unit


def test_transport_preserves_response_timeouts_and_redirect_contract(
        session: Mock, http_factory) -> None:  # fmt: skip
    with http_factory("https://example.test/root/", 5) as http:
        response = http.request("GET", "/api/ping")
        value_checks.identical(response, session.request.return_value)
        mock_checks.called_once_with(
            session.request,
            method="GET",
            url="https://example.test/root/api/ping",
            timeout=5,
            allow_redirects=False,
        )
        http.request(
            "POST",
            "chat",
            timeout=120,
            allow_redirects=True,
            json=case_data.fresh(CHAT_MESSAGE_PAYLOAD),
        )
        value_checks.equal(session.request.call_args.kwargs["timeout"], 120)
        value_checks.identical(session.request.call_args.kwargs["allow_redirects"], True)
    mock_checks.called_once_with(session.close)


def test_transport_closes_and_does_not_retry_failed_generation(
        session: Mock, http_factory, failure_factory) -> None:  # fmt: skip
    failure = failure_factory(requests.Timeout, "Model response timed out")
    session.request.side_effect = failure
    with (
        errors.expected_error(requests.Timeout) as caught,
        http_factory("http://localhost", 5) as http,
    ):
        http.request("POST", "/chat")
    value_checks.identical(caught.value, failure)
    mock_checks.called_once(session.request)
    mock_checks.called_once_with(session.close)


def test_shared_transport_does_not_leak_authentication_between_clients(
        session: Mock, api_factory, http_factory) -> None:  # fmt: skip
    with http_factory("http://localhost", 5) as http:
        authenticated = api_factory(http, api_key="test-key")
        anonymous = api_factory(http)
        authenticated.verify_authentication()
        anonymous.verify_authentication()
    value_checks.equal(
        session.request.call_args_list[0].kwargs["headers"], case_data.fresh(AUTHENTICATED_HEADERS)
    )
    value_checks.equal(
        session.request.call_args_list[1].kwargs["headers"], case_data.fresh(case_data.EMPTY_OBJECT)
    )


@pytest.mark.parametrize("slug", WORKSPACE_SLUGS_ARE_ENCODED_AS_ONE_PATH_SEGMENT_SLUG_CASES)
def test_workspace_slugs_are_encoded_as_one_path_segment(session: Mock, slug: str, api_factory,
                                                         http_factory) -> None:  # fmt: skip
    with http_factory("http://localhost", 5) as http:
        api_factory(http).get_workspace(slug)
    value_checks.equal(
        session.request.call_args.kwargs["url"],
        f"http://localhost/api/v1/workspace/{quote(slug, safe='')}",
    )


def test_workspace_creation_does_not_mutate_configuration(session: Mock, api_factory,
                                                          http_factory) -> None:  # fmt: skip
    configuration = make_workspace_template()
    with http_factory("http://localhost", 5) as http:
        api_factory(http).create_workspace("temporary", configuration)
    value_checks.equal(configuration["name"], "template")
    value_checks.equal(
        session.request.call_args.kwargs["json"],
        make_created_workspace_payload(),
    )


@title("Workspace settings update uses the authenticated API and preserves its input")
def test_workspace_update_uses_authenticated_route_and_preserves_settings(
        session: Mock, api_factory, http_factory) -> None:  # fmt: skip
    configuration = case_data.fresh(REVIEWED_CHAT_CONFIGURATION)
    with http_factory("http://localhost", 5) as http:
        response = api_factory(http, api_key="test-key").update_workspace(
            "policy/lab", configuration
        )
    value_checks.identical(response, session.request.return_value)
    mock_checks.called_once_with(
        session.request,
        method="POST",
        url="http://localhost/api/v1/workspace/policy%2Flab/update",
        headers=case_data.fresh(AUTHENTICATED_HEADERS),
        json=case_data.fresh(REVIEWED_CHAT_CONFIGURATION),
        timeout=5,
        allow_redirects=False,
    )
    value_checks.equal(configuration, case_data.fresh(REVIEWED_CHAT_CONFIGURATION))


@pytest.mark.parametrize("fail", UPLOAD_CLOSES_DOCUMENT_EVEN_WHEN_REQUEST_FAILS_FAIL_CASES)
def test_upload_closes_document_even_when_request_fails(
        session: Mock, tmp_path: Path, fail: bool, api_factory, http_factory) -> None:  # fmt: skip
    path = tmp_path / case_data.POLICY_DOCUMENT_TITLE
    path.write_text("Fictional policy", encoding="utf-8")
    documents = []

    upload = make_upload_stub(documents, fail)

    session.request.side_effect = upload
    with http_factory("http://localhost", 5) as http:
        api = api_factory(http)
        check_upload_closes_document_even_when_request_fails_outcome(api, fail, path)
    value_checks.length(documents, 1)
    value_checks.truthy(documents[0].closed)
