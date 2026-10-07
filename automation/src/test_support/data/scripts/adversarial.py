"""Isolated pytest source templates; no application/model execution."""

COLLECTION_MAKEPYFILE_SOURCE = """
        import pytest
        @pytest.mark.rag
        @pytest.mark.adversarial
        def test_attack(adversarial_case, generation_model, rag_iteration): pass
    """
