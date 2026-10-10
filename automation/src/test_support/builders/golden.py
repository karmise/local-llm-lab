"""The reviewed golden policy dataset and captured samples bound to its paid-leave case."""

from collections.abc import Sequence
from typing import Any

from llm_testkit.datasets.golden import load_golden_dataset
from llm_testkit.observation.evaluation_sample import build_sample
from test_support.data.common import TEST_MODEL
from test_support.paths import AUTOMATION_ROOT

TEST_DATA = AUTOMATION_ROOT / "test_data"
POLICY_FILE = TEST_DATA / "company-policy.txt"
GOLDEN_DATASET_FILE = TEST_DATA / "golden-policy.json"
GOLDEN_DATASET = load_golden_dataset(GOLDEN_DATASET_FILE, POLICY_FILE)
PAID_LEAVE = next(case for case in GOLDEN_DATASET.cases if case.id == "paid_leave")
CAPTURE_ID = "a" * 32
POLICY_DOCUMENT = f"automation-{CAPTURE_ID}-company-policy.txt"


def make_paid_leave_sample(contexts: Sequence[str] | None = None) -> dict[str, Any]:
    """A captured paid-leave sample whose answer equals the golden reference.

    By default the model saw one context: the policy document with its AnythingLLM metadata header.
    """
    if contexts is None:
        contexts = [
                f"<document_metadata>\nsourceDocument: {POLICY_DOCUMENT}\n</document_metadata>\n" +
                POLICY_FILE.read_text()]
    blocks = "\n".join(f"[CONTEXT {i}]:\n{context}\n[END CONTEXT {i}]" for i, context in enumerate(contexts))
    system = f"[LLM_TESTKIT_CAPTURE:{CAPTURE_ID}]\n{blocks}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": PAID_LEAVE.question}]
    request = {"model": TEST_MODEL, "stream": False, "messages": messages}
    observation = {"schema_version": 1, "boundary": "ollama-sdk-chat", "request": request}
    sample = build_sample(
            observation, question=PAID_LEAVE.question, answer=PAID_LEAVE.reference, reference=PAID_LEAVE.reference,
            expected_model=TEST_MODEL, capture_id=CAPTURE_ID)
    sample["response_sources"] = [{"title": POLICY_DOCUMENT, "text": context} for context in contexts]
    return sample
