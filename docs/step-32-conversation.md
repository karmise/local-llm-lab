# Conversational policy assistant

The application has two reviewed workspace profiles:

- `config/workspace.json`: document-only Query mode for the existing RAG, golden,
  prompt regression and quality scenarios.
- `config/conversation-workspace.json`: Chat mode for greetings, everyday
  conversation and mixed requests containing policy questions. The manual
  `company-policy-lab` workspace uses this profile.

The conversational contract permits brief greetings and ordinary suggestions
about everyday activities. Company policy claims require the supplied documents.
Unknown policies, including policies of another company, must be acknowledged as
unavailable. A mixed request should answer each part without transferring one
company's rules to another. Responses follow the user's language; automated
acceptance scenarios use English.

AnythingLLM 1.16.2 performs a vector search for each ordinary Chat request when
the workspace contains indexed documents. This profile controls answer behavior;
it does not implement conditional retrieval. A short greeting or an absence of
citations in its final text is not evidence that retrieval was skipped. Automatic
agent mode is a different execution path and is outside this profile's contract.

The conversational profile keeps at most one previous history entry in the
application prompt to limit unrelated context. Each automated scenario receives
a fresh indexed workspace, so one scenario cannot provide another's answer.
The policy document, generation model and retrieval settings remain explicit.

To reproduce the manual configuration, open the workspace settings and apply
the conversational JSON fields. Updating settings does not require deleting
documents or chat history. Reload the UI after changing the mode or prompt.

These curated acceptance checks exercise response behavior, not universal
conversational quality, medical advice safety or a guarantee against hallucination.

## Acceptance scenarios

The five reviewed cases in `automation/test_data/conversation-policy.json` cover:

1. A greeting: a short reply without an unsolicited policy or document summary.
2. An everyday walk suggestion: a relevant reply without invented weather.
3. Northern Lighthouse leave: 23 working days per year, at least 12 calendar
   days' notice, direct manager approval, and the uploaded source.
4. HarborWorks leave: no approved information and no invented entitlement.
5. A mixed request: greeting, walking, Northern Lighthouse leave and an explicit
   limitation for HarborWorks, with the company claims kept separate.

Run from `automation` with its virtual environment active:

```bash
python -m pytest tests/conversation --run-conversation --rag-model qwen3.5:4b \
  --junitxml=reports/conversation/junit.xml --alluredir=reports/conversation/allure
```

Omit `--alluredir` without reporting dependencies. For a smaller first run add
`-k greeting`. Each selected case/model/repetition makes one answer request with
no retry. Five cases on one model make five sequential generations; no judge
calls are added. A failed answer stays failed. To compare models, add another
`--rag-model qwen2.5:7b`; this doubles the generation budget.

The dataset loader validates its policy checksum, IDs, regex rules, source
anchors, word budgets and reviewed reference examples before generation.
References demonstrate acceptable content, not exact wording. Shared assertions
check the completed final answer, all required intents, prohibited content and
source passages for policy answers. Word limits of 40/90/140/160 are explicit
conversational acceptance budgets, not semantic-quality scores. Regex checks have
limited paraphrase coverage and do not prove that every possible assertion is true.

Fixtures own fresh indexed workspaces and verify cleanup. Tests contain scenario
calls and shared assertions; Allure steps stay in clients/assertions. Each Allure
test has a case-specific title. JUnit records case/catalog identity, model digest,
policy hash and actual workspace settings. Stability summaries keep different
conversation cases separate and reject changed or missing catalog fingerprints.

`--capture-rag` is rejected for enabled conversation checks: greetings and small
talk are not document-grounded evaluation samples. Returned `sources` may still
contain policy matches for a greeting; the final answer must not dump them.
These supplemental checks are not included in the existing 13-requirement
IQ/OQ/PQ qualification protocol; they do not close that protocol's PQ matrix.
