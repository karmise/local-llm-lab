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
