# Step 13: Expanded Faithfulness Controls

## New scenarios

The hand-labelled catalog now contains six cases. Three additional controls test
paraphrases, answer omissions and the distinction between working/calendar days:

| Control | Expected faithfulness | What it checks |
| --- | --- | --- |
| `supported_paraphrase` | 1.0 | Correct policy facts expressed differently, with numbers written as words |
| `incomplete_supported_answer` | 1.0 | Only the paid-leave allowance is answered; the notice period is omitted |
| `wrong_day_type` | 0.0 | 23 calendar days are incorrectly substituted for 23 working days |

The incomplete answer deliberately receives an expected score of 1.0. Its one
claim is supported by the context. Faithfulness does not require the answer to
address every part of the question. Keep the existing paid-leave scenario's
required-fact assertions alongside this metric; a faithful answer can still be
incomplete.

## Targeted run

From `automation`, using the saved step 10 sample and installed evaluation extra:

```bash
.venv/bin/python -m llm_testkit.evaluation.calibration \
  reports/rag-samples/<capture-id>.json \
  --controls test_data/faithfulness-controls.json \
  --control supported_paraphrase \
  --control incomplete_supported_answer \
  --control wrong_day_type \
  --judge-model qwen3.5:4b \
  --output reports/expanded-controls-<capture-id>.json
```

Catalog loading validates all labels and required context fragments, independent
of batch size. Selection and execution each enforce a maximum of three controls
per run. Selecting the entire six-case catalog, either implicitly or explicitly,
fails before model calls. A single control can be evaluated with one `--control`
option. Existing controls remain available as a separate small batch.

No model retries, thinking, application generations or new Python dependencies
are introduced. Expected labels never enter judge prompts. Reports retain each
synthetic response, extracted claims, verdicts/reasons, sample/control checksums
and model settings, including errors or mismatches.

## Interpretation and next step

These controls broaden the diagnostic sample but do not establish statistically
reliable judge calibration. Regex annotations accept numeric and written-number
paraphrases for these simple facts, while requiring the expected claim count and
distinct matches. Unanticipated extraction or wording causes a mismatch that
should be reviewed rather than silently accepted.

The next framework step should combine independent quality dimensions in a
report: required-fact coverage and source checks from deterministic assertions,
alongside model-based faithfulness. Introduce thresholds only after broader
labelled examples and agreement measurements.

## Verified run

On 2026-10-01, Qwen3.5 4B matched all three new hand-labelled controls:
paraphrase 1.0, incomplete supported answer 1.0 and wrong day type 0.0.
The six sequential judge calls took 23.40 seconds according to Ollama server
durations. The saved application sample was reused without regeneration.

All 64 framework unit checks passed. New coverage verifies loading a six-case
catalog, rejecting implicit or explicit oversized batches before model calls,
and rejecting an incomplete answer through required-fact assertions even when
its faithfulness score is 1.0. Raw verdicts remain in the ignored local report
`automation/reports/step-13-judge-controls.json`.
