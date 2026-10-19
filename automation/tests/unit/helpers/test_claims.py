"""Claims: an answer is split into its sentences in code, unchanged, so no model paraphrases what is checked."""

import pytest

from helpers.claims import split_claims
from reporting.steps import title

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("text", "claims"), [
        pytest.param("One fact. Another fact.", ["One fact.", "Another fact."], id="two-sentences"),
        pytest.param("Is it allowed? No! Ask the manager.", ["Is it allowed?", "No!", "Ask the manager."], id="marks"),
        pytest.param(
        'The policy states: "Hotels need a receipt." Ask finance.',
        ['The policy states: "Hotels need a receipt."', "Ask finance."], id="closing-quote"),
        pytest.param("The limit is KGS 8400 per night.", ["The limit is KGS 8400 per night."], id="one-sentence"),
        pytest.param("Version 2.5 applies. Done.", ["Version 2.5 applies.", "Done."], id="decimal-number"),
        pytest.param("See e.g. section 2 for details.", ["See e.g. section 2 for details."], id="lowercase-after-dot"),
        pytest.param("  Leading and trailing spaces.  ", ["Leading and trailing spaces."], id="surrounding-space"),
        pytest.param("Line one.\nLine two.", ["Line one.", "Line two."], id="newline")])
@title("An answer is split into its sentences, each kept exactly as written [{param_id}]")
def test_split_claims(text, claims):
    assert split_claims(text) == claims


@pytest.mark.parametrize("text", ["", "   ", "...", "1. 2."])
@title("Text without a word yields no claim [{text!r}]")
def test_split_claims_without_words(text):
    assert split_claims(text) == []
