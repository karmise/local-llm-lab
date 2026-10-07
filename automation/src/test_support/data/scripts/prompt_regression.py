"""Isolated pytest source templates; no application/model execution."""

COLLECTION_MAKEPYFILE_SOURCE = """
        import pytest
        @pytest.mark.rag
        @pytest.mark.prompt_regression
        def test_prompt(prompt_variant, golden_case, generation_model, rag_iteration):
            assert golden_case.id == 'carryover_limit'
            assert prompt_variant.id in ('baseline', 'grounded_v2')
    """
