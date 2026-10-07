"""Named parameter cases for http_clients scenarios."""

WORKSPACE_SLUGS_ARE_ENCODED_AS_ONE_PATH_SEGMENT_SLUG_CASES = [
    "a/b",
    "a b",
    "x?query#fragment",
    "non-ascii-\N{SNOWMAN}",
]

UPLOAD_CLOSES_DOCUMENT_EVEN_WHEN_REQUEST_FAILS_FAIL_CASES = [False, True]
