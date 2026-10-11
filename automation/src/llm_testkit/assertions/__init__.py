"""Reusable API, answer, quality and UI checks; clients and page objects do not assert outcomes.

Import as ``from llm_testkit import assertions`` and call ``assertions.assert_...``.
"""

from llm_testkit.assertions.answers import (
        assert_adversarial_answer, assert_adversarial_context, assert_attack_exposure, assert_bias_answer,
        assert_completed_answer, assert_conversation_answer, assert_document_sources, assert_golden_answer,
        assert_golden_text, assert_missing_policy_answer, assert_missing_policy_information, assert_rag_answer,
        assert_required_facts)
from llm_testkit.assertions.api import (
        assert_api_key_accepted, assert_api_key_rejected, assert_created_workspace, assert_document_folder_absent,
        assert_embeddings_updated, assert_model_available, assert_models_available, assert_online,
        assert_operation_success, assert_search_contains, assert_supported_runtime, assert_uploaded_document,
        assert_workspace_absent, assert_workspace_document_attached, assert_workspace_matches)
from llm_testkit.assertions.fields import (
        assert_field_contains, assert_field_equals, assert_field_length, assert_field_starts_with, assert_field_type,
        assert_json_object, assert_status_code)
from llm_testkit.assertions.quality import (
        assert_benchmark_case, assert_benchmark_report, assert_calibration_result, assert_performance_batch,
        assert_quality_dimension, assert_quality_report, assert_quality_score)
from llm_testkit.assertions.ui import (
        assert_ui_completed_answer, assert_ui_document_source, assert_ui_history_preserved, assert_ui_policy_answer,
        assert_ui_question_visible, assert_ui_source_content)

__all__ = [
        "assert_status_code", "assert_json_object", "assert_field_type", "assert_field_equals", "assert_field_length",
        "assert_field_contains", "assert_field_starts_with", "assert_online", "assert_api_key_accepted",
        "assert_api_key_rejected", "assert_created_workspace", "assert_workspace_matches", "assert_workspace_absent",
        "assert_operation_success", "assert_uploaded_document", "assert_embeddings_updated",
        "assert_workspace_document_attached", "assert_search_contains", "assert_document_folder_absent",
        "assert_model_available", "assert_models_available", "assert_supported_runtime", "assert_completed_answer",
        "assert_rag_answer", "assert_required_facts", "assert_golden_answer", "assert_adversarial_context",
        "assert_golden_text", "assert_document_sources", "assert_conversation_answer",
        "assert_missing_policy_information", "assert_missing_policy_answer", "assert_attack_exposure",
        "assert_adversarial_answer", "assert_bias_answer", "assert_quality_score", "assert_calibration_result",
        "assert_benchmark_report", "assert_benchmark_case", "assert_quality_dimension", "assert_quality_report",
        "assert_performance_batch", "assert_ui_question_visible", "assert_ui_completed_answer",
        "assert_ui_policy_answer", "assert_ui_document_source", "assert_ui_source_content",
        "assert_ui_history_preserved"]
