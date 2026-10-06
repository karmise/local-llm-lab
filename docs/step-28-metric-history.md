# Metric history and baseline monitoring

`llm_testkit.reporting.drift` records immutable snapshots from validated sample, faithfulness, correctness and relevance evidence. It stores all four scores, deterministic fact/source outcomes, model digest, policy/dataset hashes, the actual prompt/configuration, judge configuration/version fingerprint and evidence checksums. Capture markers are removed from the prompt fingerprint. A duplicate captured answer is rejected as a new observation.

From `automation`:

```bash
.venv/bin/python -m llm_testkit.reporting.drift record \
  --sample reports/evidence/sample.json --faithfulness reports/evidence/faithfulness.json \
  --correctness reports/evidence/correctness.json --relevance reports/evidence/relevance.json
.venv/bin/python -m llm_testkit.reporting.drift compare \
  reports/history/<baseline>.json reports/history/<current>.json \
  --maximum-drop 0.05 --output reports/drift/comparison.json
```

Select a reviewed baseline explicitly; recording a run never promotes it automatically. The comparison exits nonzero for a score drop greater than the tolerance, failed acceptance, duplicates or incomparable conditions. Judge, prompt, dataset, policy and retrieval changes make observations incomparable. A generation-model name/digest change is allowed and disclosed, so a candidate can be compared against the same baseline.

History lives under ignored `reports/history`. Keep it as a retained artifact between runs. Invoke the comparison in an evidence-producing pipeline after each evaluation; it makes zero model calls. Monitoring here means checking subsequent observations against the declared baseline. Scheduling model generation, collecting production traffic and statistically detecting distribution drift are outside this local implementation. A one-case drop is a regression signal, not proof of population-level drift or its cause.
