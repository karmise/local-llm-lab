"""Named inputs and catalog mutations for judge validation scenarios."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTROL_IDS = (
        "receipt_condition_supported", "receipt_condition_contradicted", "manager_policy_paraphrase",
        "manager_wrong_role", "relevant_context_first", "relevant_context_second", "all_reference_facts_retrieved",
        "notice_fact_not_retrieved")
INVALID_CATALOG_CHANGES = (
        "policy_hash", "dataset_hash", "duplicate_id", "reference", "invented_context", "boolean_score",
        "boolean_verdict", "score_label_disagreement", "empty_pattern", "invalid_metric", "extra_label_field")
RECALL_STATEMENTS = (
        "Each employee receives 23 working days of paid leave per year.",
        "A leave request must be submitted at least 12 calendar days before leave starts.")
