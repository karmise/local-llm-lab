"""Named payload construction for provenance scenarios."""


def make_configuration_with_capture_suffix(marker):
    """Build input for test_capture_suffix_is_incidental_and_configuration_is_not_mutated."""
    return {"openAiPrompt": "Policy\n" + marker, "options": {"topN": 4}}


def make_configuration_with_capture_prefix(marker):
    """Build input for test_capture_suffix_is_incidental_and_configuration_is_not_mutated."""
    return {"openAiPrompt": marker + "\nPolicy"}
