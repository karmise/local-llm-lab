# Adversarial inputs and hallucination smoke checks

Six reviewed inputs in `test_data/adversarial-policy.json` exercise instruction override, role spoofing, fabricated source authority, invented benefits, instructions embedded in a document, and an untrusted contradictory note. Each attack binds to the current golden dataset and reuses its original reference, required facts, forbidden facts and source anchors. Attacker text never becomes an expected answer.

Two scenarios append a clearly untrusted note to a temporary policy copy. The canonical policy and its golden checksum stay unchanged; the actual uploaded copy gets its own recorded checksum. Function-scoped application resources and existing cleanup remain in use. JUnit records attack identity/category, catalog version/hash and original policy hash; optional capture retains these properties. Stability groups attacks separately.

From `automation`, run one user-input attack first:

```bash
.venv/bin/python -m pytest tests/test_adversarial.py --run-adversarial \
  --rag-model qwen3.5:4b -k user_override \
  --junitxml=reports/adversarial/runs.xml
```

That selection makes one generation call. A full matrix uses six attacks × two models = 12 calls before repetitions. Tests skip by default. There are no model retries or automatic repairs of a failed answer.

Document attacks additionally require the capture Compose overlay and `--capture-rag`. They verify the full appended attack text in the actual captured model context. A test cannot pass simply because retrieval never exposed the attack. A source citation alone is insufficient to establish exposure. User-input attacks do not require capture.

Shared assertions check the final answer after removing thinking text, require the original policy facts/sources, and reject attack markers or labelled poisoned values. The invented-benefit scenario also uses the golden missing-information rules, which reject monetary amounts. These conservative lexical rules can flag quoted attack text even when the model refuses it. They can also miss semantic contradictions or novel paraphrases: they are targeted regression smoke checks, not a comprehensive security assessment or hallucination detector. Reference-based correctness and faithfulness evaluations provide separate semantic signals.

Captured attack questions differ from canonical golden questions. The ordinary correctness/relevance evaluator deliberately refuses to bind those samples as ordinary golden evidence; do not relabel attack samples to bypass provenance checks. Extend the evaluation schema explicitly if semantic scoring of attacks becomes necessary.
