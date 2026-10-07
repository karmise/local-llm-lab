"""Lazy optional types keep unit collection independent of evaluation extras."""

import pytest


def load_ollama_judge():
    pytest.importorskip("ragas", reason="Install evaluation dependencies")
    from llm_testkit.evaluation.ollama_judge import OllamaJudge

    return OllamaJudge
