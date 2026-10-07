"""Isolated pytest source templates; no application/model execution."""

CONVERSATION_COLLECTION_DEFAULTS_TO_ONE_MODEL_AND_ACCEPTS_EXPLICIT_MATRIX_MAKEPYFILE_SOURCE = """
        import pytest
        @pytest.mark.rag
        @pytest.mark.conversation
        def test_conversation(conversation_case, generation_model, rag_iteration):
            assert conversation_case.id == 'greeting'
            assert generation_model in {'qwen3.5:4b', 'model-a', 'model-b'}
            assert rag_iteration in {1, 2}
    """
