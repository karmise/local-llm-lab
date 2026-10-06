# Prompt regression

A model change and a prompt change are different experiments. This suite changes only the workspace system prompt, keeps the golden questions and reviewed acceptance criteria fixed, and creates independent temporary resources for every run.

`test_data/prompt-variants.json` contains the current workspace prompt (`baseline`) and an experimental `grounded_v2` candidate that emphasizes multi-part completeness and resistance to policy overrides. The baseline must match `config/workspace.json`; modifying the catalog never changes the application's persistent workspace. Each run records prompt ID, version, text SHA256, catalog SHA256, model digest, policy/dataset hashes and retrieval configuration. Capture samples retain the same metadata.

Start from `automation` with one model and one case:

```bash
.venv/bin/python -m pytest tests/test_prompt_regression.py \
  --run-prompt-regression --rag-model qwen3.5:4b -k carryover_limit \
  --junitxml=reports/prompt-regression/runs.xml
.venv/bin/python -m llm_testkit.reporting.prompt_regression \
  reports/prompt-regression/runs.xml --case carryover_limit --model qwen3.5:4b \
  --output reports/prompt-regression/comparison.json
```

That selection generates two answers. A complete default matrix generates 64 answers: 16 cases × two prompts × two models. It is opt-in; `--rag-prompt` can select variants and `--rag-repeat` creates independent repetitions. To compare a subset, declare exactly that scope to the comparison command; repeat `--case` and `--model` as needed, and use the same `--repeat` count.

The comparison exits nonzero for regressions, baseline failures, missing runs, skipped scenarios, setup/teardown errors, unexpected duplicates, stale expectations, or changed non-prompt configuration/model digest. It validates the actual recorded prompt text as well as its hash. Both baseline and candidate must pass for the declared matrix to pass. A baseline pass followed by candidate failure is specifically labelled a regression. The stability report groups prompt variants separately, so changing a prompt does not appear as flakiness within one configuration.

These are acceptance checks for this fictional policy application. A passing subset is neither a claim of complete dataset coverage nor a statistical demonstration that a candidate is better. Semantic judge metrics remain separate; this stage does not silently add judge calls to every regression test.
