"""Isolated pytest source templates; no application/model execution."""

OPT_IN_SCENARIOS_SKIP_BEFORE_RESOLVING_EXTERNAL_FIXTURES_MAKEPYFILE_SOURCE = """
        import pytest
        @pytest.mark.ui
        def test_ui(missing_browser): pass
        @pytest.mark.browser
        def test_browser(missing_browser): pass
        @pytest.mark.live_quality
        def test_live(missing_judge): pass
        @pytest.mark.golden
        def test_golden(missing_model): pass
        @pytest.mark.conversation
        def test_conversation(missing_model): pass
    """

GOLDEN_COLLECTION_FORMS_CASE_MODEL_REPEAT_MATRIX_MAKEPYFILE_SOURCE = """
        import pytest
        @pytest.mark.rag
        @pytest.mark.golden
        def test_golden(golden_case, generation_model, rag_iteration):
            assert golden_case.id == 'travel_allowance'
            assert generation_model in {'model-a', 'model-b'}
            assert rag_iteration in {1, 2}
    """

DESELECTED_LIVE_TESTS_DO_NOT_VALIDATE_LIVE_OPTIONS_MAKEPYFILE_SOURCE = """
        import pytest
        @pytest.mark.unit
        def test_offline(): pass
        @pytest.mark.ui
        def test_ui(missing_browser): pass
        @pytest.mark.live_quality
        def test_live(missing_judge): pass
    """

MODEL_MATRIX_DEDUPLICATES_NAMES_AND_RETAINS_INDEPENDENT_REPETITIONS_MAKEPYFILE_SOURCE = """
        import pytest
        @pytest.mark.rag
        def test_rag(generation_model, rag_iteration):
            assert generation_model in {"model-a", "model-b"}
            assert rag_iteration in {1, 2}
    """

LIVE_QUALITY_BUDGET_APPLIES_TO_SELECTED_MATRIX_MAKEPYFILE_SOURCE = """
        import pytest
        @pytest.mark.rag
        @pytest.mark.live_quality
        def test_live(generation_model, rag_iteration): pass
    """

MODEL_MATRIX_DOES_NOT_REQUIRE_AN_UNUSED_ITERATION_FIXTURE_MAKEPYFILE_SOURCE = """
        import pytest
        @pytest.mark.rag
        def test_model_only(generation_model):
            assert generation_model == "model-a"
    """
