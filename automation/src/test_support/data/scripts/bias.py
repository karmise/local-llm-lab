"""Isolated pytest source templates; no application/model execution."""

SELECTION_MAKEPYFILE_SOURCE = """
        import pytest
        @pytest.mark.bias
        @pytest.mark.rag
        def test_pair(bias_case,generation_model,rag_iteration):
            assert bias_case.pair_id == 'gender_carryover'
    """
