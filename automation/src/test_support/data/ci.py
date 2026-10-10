"""CI scope and negative provenance cases, kept outside test scenarios."""

REVISION = "a" * 40
OTHER_REVISION = "b" * 40
PRESETS = (("smoke", "primary", 2, 19), ("curated", "primary", 4, 43), ("smoke", "comparison", 4, 32),
        ("curated", "comparison", 8, 80))
INVALID_IDENTITY = ("revision", "source", "profile", "baseline", "plan", "generation_weights", "judge_weights", "test")
MODEL_NAMES = ("qwen3.5:4b", "bge-m3:567m")
