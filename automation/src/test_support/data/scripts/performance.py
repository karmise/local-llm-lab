"""Isolated pytest source templates; no application/model execution."""

SELECTION_MAKEPYFILE_SOURCE = """
        import pytest
        @pytest.mark.performance
        def test_health(missing_api): pass
        @pytest.mark.performance
        @pytest.mark.rag
        def test_rag(missing_model): pass
    """


INCOMPATIBLE_CAPTURE_MODE_FAILS_BEFORE_EXTERNAL_SETUP_MAKEPYFILE_SOURCE = """
        import pytest
        @pytest.mark.performance
        @pytest.mark.rag
        def test_batch(missing_external_service): pass
    """
