# Context precision and recall

This stage measures retrieval separately from answer correctness. A faithful answer can still omit facts, and a correct answer can be accompanied by irrelevant retrieved chunks.

We use RAGAS `ContextPrecisionWithReference` (average precision over ranked chunks) and `ContextRecall` (the fraction of reference statements supported by the captured context). Neither metric is keyword overlap. A local structured judge supplies the verdicts. The evaluator recomputes both scores, verifies binary verdicts and reasons, and checks that the recall decomposition includes the golden case's labelled facts. This last check is a conservative regex guard, not proof of complete semantic decomposition.

Run from `automation` after installing the evaluation extra:

```bash
.venv/bin/python -m llm_testkit.evaluation.relevance reports/rag-samples/<sample>.json \
  --case paid_leave --output reports/relevance/paid-leave.json
```

The saved question and reference must match the current source-bound golden dataset. The evaluator uses the actual captured contexts in their original order. It allows one to four contexts, never truncates, performs at most five sequential judge calls (one per chunk plus one recall call), and never retries. Errors retain raw calls. Each metric remains an exploratory measurement until explicit quality gates are supplied.

Add `--relevance-report reports/relevance/paid-leave.json` to the saved quality test alongside the existing faithfulness and correctness evidence. Both context dimensions appear independently in Allure, including invalid-evidence errors. Unit tests run the real RAGAS metrics with mocked judges and exercise missing facts, invalid verdicts, provenance mismatches and call limits.

Definitions: [RAGAS context precision](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) and [context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/).
