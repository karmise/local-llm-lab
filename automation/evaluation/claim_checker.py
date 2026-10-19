"""Check each sentence of an answer against the retrieved context with a dedicated grounding model.

Code, not a model, splits the answer into claims (helpers/claims.py), so the checker never grades its own paraphrase.
"""

from typing import Any

from api.ollama import OllamaClient

CHECKER_MODEL = "bespoke-minicheck:7b"
MAX_CLAIMS = 6


class ClaimChecker:
    """One bounded, non-streaming Yes/No call per claim, recorded with its prompt and raw response."""
    def __init__(self, client: OllamaClient, model: str, timeout: float = 300, *, max_calls: int = MAX_CLAIMS) -> None:
        if type(max_calls) is not int or not 1 <= max_calls <= MAX_CLAIMS:
            raise ValueError(f"Claim checker budget must be an integer between one and {MAX_CLAIMS}")
        self.client = client
        self.model = model
        self.timeout = timeout
        self.max_calls = max_calls
        self.options = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 8}
        self.calls: list[dict[str, Any]] = []

    def check(self, document: str, claim: str) -> int:
        """1 when ``document`` supports ``claim``, 0 when it does not."""
        if len(self.calls) >= self.max_calls:
            raise ValueError(f"Claim checker budget exceeded (maximum: {self.max_calls})")
        prompt = f"Document: {document}\nClaim: {claim}"
        call: dict[str, Any] = {"claim": claim, "prompt": prompt}
        self.calls.append(call)
        response = self.client.chat(model=self.model, prompt=prompt, options=self.options, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        call["ollama_response"] = payload
        if payload.get("done") is not True or payload.get("done_reason") != "stop":
            raise ValueError("Claim check is incomplete or truncated")
        if payload.get("model") != self.model:
            raise ValueError("Claim checker returned an unexpected model")
        answer = payload["message"]["content"].strip()
        if answer not in ("Yes", "No"):
            raise ValueError(f"Claim checker must answer Yes or No, got {answer[:40]!r}")
        call["verdict"] = int(answer == "Yes")
        return call["verdict"]
