# Step 12: Hand-Labelled Judge Controls

## Purpose

Check whether the local faithfulness judge distinguishes supported facts from
contradictions and unsupported claims. These are synthetic answer controls,
not new AnythingLLM responses or evidence of application accuracy. The source
sample supplies the actual question and context captured in step 10.

The English control dataset is `automation/test_data/faithfulness-controls.json`.
It contains explicit answers, expected scores and hand-labelled claim patterns:

| Control | Expected claims | Expected faithfulness |
| --- | --- | --- |
| `supported_answer` | 23 working days and at least 12 calendar days' notice, both supported | 1.0 |
| `wrong_numbers` | 30 working days and permission to submit only 2 calendar days before leave, both contradicted | 0.0 |
| `unsupported_gym_amount` | 23 working days supported; KGS 5000 gym reimbursement unsupported | 0.5 |

The wrong-notice control explicitly claims that two days meets the requirement.
A weaker statement such as "at least two days" can be implied by "at least twelve
days" and is ambiguous as a faithfulness control. Use clear contradictions when
hand-labelling expected verdicts.

## Run

From `automation`, with the [evaluation extra](step-11-faithfulness.md) installed:

```bash
.venv/bin/python -m llm_testkit.evaluation.calibration \
  reports/rag-samples/<capture-id>.json \
  --controls test_data/faithfulness-controls.json \
  --control supported_answer --control wrong_numbers --control unsupported_gym_amount \
  --judge-model qwen3.5:4b \
  --output reports/judge-controls-<capture-id>.json
```

To repeat just one control, add `--control wrong_numbers`. The option can be
repeated; duplicate selections are deduplicated and unknown identifiers fail
before model calls. The catalog can contain more controls, but a run is limited
to one through three explicitly selected controls, processed
sequentially with up to two judge calls per control. No application answer is
generated. Normal pytest runs remain free of judge requests.

Since [step 13](step-13-expanded-judge-controls.md), the catalog contains six
cases. Omitting `--control` for the full catalog fails before contacting Ollama.
Small catalogs of up to three controls can still be run without explicit selection.

## Checks and reports

The service first validates the captured sample and required policy fragments,
then validates control IDs, scores and hand-labelled verdicts. This dataset
applies specifically to the Northern Lighthouse policy; changing the corpus
requires reviewing the labels and context anchors.

Each control receives a fresh judge with the step 11 settings, thinking disabled
and no retries. Expected labels are used only by the local checker; they are
never included in the judge prompt. The original sample is preserved, and only
the synthetic response is substituted for evaluation.

Centralized assertions compare the score and every expected claim verdict.
A matching average with reversed verdicts fails. Missing claims, merged claims,
duplicate matches or unexpected statement counts fail the control check.
Regex patterns allow paraphrases within these simple controls; they are not a
general semantic annotation matcher. Inspect the retained prompts and verdicts
when a mismatch occurs.

Per-control statuses distinguish `matched`, `mismatch` and `error`. A mismatch
means valid model output disagreed with the hand labels. An error means evaluation
could not complete. Subsequent controls continue so the report preserves all
results. The command exits 0 only if every selected control matched; otherwise
it exits 1. This is a gate for control agreement, not a production RAG threshold.

Reports retain answers, labels, statements, reasons, raw judge responses,
model digest/settings, RAGAS version, selected IDs and checksums of the source
sample and controls. Reports remain local, ignored and protected against
overwriting. No new Python dependencies are added.

## Interpretation

Three controls are a diagnostic sanity check, not statistical calibration or
proof of reliable judgement. A model can pass them and still fail on paraphrases,
implicit claims, adversarial context or longer answers. Keep deterministic
scenario assertions and review failures. Broader hand-labelled cases and judge
agreement measurements are needed before choosing quality thresholds.

The initial local run matched the three expected scores (1.0, 0.0 and 0.5).
The notice contradiction was then clarified and verified separately using
targeted control selection. Raw reports record the exact wording evaluated.

On 2026-10-01, all 61 framework unit checks passed. The initial three-control
run used six calls and 32.18 seconds of Ollama server time. The targeted notice
verification used two additional calls; no application answers were regenerated.
Unit coverage checks reversed verdicts with matching averages, missing/merged
claims, invalid labels/context, targeted selection, call budgets and preservation
of errors and mismatches while subsequent controls continue.
