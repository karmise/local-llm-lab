"""The reviewed golden policy dataset and captured samples bound to its cases."""

from collections.abc import Sequence
from typing import Any

from llm_testkit.datasets.golden import GoldenCase, load_golden_dataset
from llm_testkit.observation.evaluation_sample import build_sample
from test_support.builders.identities import TEST_MODEL
from test_support.paths import AUTOMATION_ROOT

TEST_DATA = AUTOMATION_ROOT / "test_data"
POLICY_FILE = TEST_DATA / "company-policy.txt"
GOLDEN_DATASET_FILE = TEST_DATA / "golden-policy.json"
GOLDEN_DATASET = load_golden_dataset(GOLDEN_DATASET_FILE, POLICY_FILE)
CASES = {case.id: case for case in GOLDEN_DATASET.cases}
PAID_LEAVE = CASES["paid_leave"]
CAPTURE_ID = "a" * 32
POLICY_DOCUMENT = f"automation-{CAPTURE_ID}-company-policy.txt"


def policy_context() -> str:
    """The policy document as AnythingLLM passes it to the model, with its metadata header."""
    return f"<document_metadata>\nsourceDocument: {POLICY_DOCUMENT}\n</document_metadata>\n" + POLICY_FILE.read_text()


def make_case_sample(case: GoldenCase, *, model: str = TEST_MODEL, contexts: Sequence[str] | None = None) -> dict[str,
        Any]:
    """A captured sample for a golden case whose answer equals the reference; the context defaults to the policy."""
    contexts = [policy_context()] if contexts is None else contexts
    blocks = "\n".join(f"[CONTEXT {i}]:\n{context}\n[END CONTEXT {i}]" for i, context in enumerate(contexts))
    system = f"[LLM_TESTKIT_CAPTURE:{CAPTURE_ID}]\n{blocks}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": case.question}]
    request = {"model": model, "stream": False, "messages": messages}
    observation = {"schema_version": 1, "boundary": "ollama-sdk-chat", "request": request}
    sample = build_sample(
            observation, question=case.question, answer=case.reference, reference=case.reference, expected_model=model,
            capture_id=CAPTURE_ID)
    sample["response_sources"] = [{"title": POLICY_DOCUMENT, "text": context} for context in contexts]
    return sample


def make_paid_leave_sample(contexts: Sequence[str] | None = None) -> dict[str, Any]:
    """A captured paid-leave sample; see make_case_sample."""
    return make_case_sample(PAID_LEAVE, contexts=contexts)
