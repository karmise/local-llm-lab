"""Named parameter cases for http_clients scenarios."""

WORKSPACE_SLUGS_ARE_ENCODED_AS_ONE_PATH_SEGMENT_SLUG_CASES = ["a/b", "a b", "x?query#fragment", "non-ascii-\N{SNOWMAN}"]

UPLOAD_CLOSES_DOCUMENT_EVEN_WHEN_REQUEST_FAILS_FAIL_CASES = [False, True]

# Input for test_transport_preserves_response_timeouts_and_redirect_contract
CHAT_MESSAGE_PAYLOAD = {"message": "hello"}

# Input for test_shared_transport_does_not_leak_authentication_between_clients
AUTHENTICATED_HEADERS = {"Authorization": "Bearer test-key"}

# Input for test_workspace_update_uses_authenticated_route_and_preserves_settings
REVIEWED_CHAT_CONFIGURATION = {"chatMode": "chat", "openAiPrompt": "Reviewed prompt"}
