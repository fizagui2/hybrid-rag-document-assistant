from types import SimpleNamespace

import pytest

from src.generation.citations import extract_claim_citations, parse_citations, verify_citations
from tests.conftest import requires_groq


def _chunk(text):
    return SimpleNamespace(text=text)


# --- parse_citations: pure regex logic, no model needed ---

def test_parse_citations_plain_ascii_brackets():
    assert parse_citations("Paris is the capital [1].") == [1]


def test_parse_citations_multiple_bracket_pairs():
    assert parse_citations("Supported by two sources[1][2].") == [1, 2]


def test_parse_citations_comma_separated_within_one_bracket():
    assert parse_citations("Supported by both [1, 2].") == [1, 2]


def test_parse_citations_fullwidth_cjk_brackets():
    # observed in real Groq output despite the prompt asking for plain [1]
    assert parse_citations("巴黎是法国的首都【1】。") == [1]


def test_parse_citations_no_citation():
    assert parse_citations("This sentence cites nothing.") == []


# --- extract_claim_citations: pure logic, no model needed ---

def test_extract_claim_citations_pairs_sentence_with_correct_chunk():
    chunks = [_chunk("Paris is the capital of France."), _chunk("The Eiffel Tower is in Paris.")]
    answer = "The capital of France is Paris [1]. It is home to the Eiffel Tower [2]."

    pairs = extract_claim_citations(answer, chunks)

    assert len(pairs) == 2
    assert pairs[0].citation_number == 1
    assert pairs[0].chunk_text == "Paris is the capital of France."
    assert pairs[1].citation_number == 2
    assert pairs[1].chunk_text == "The Eiffel Tower is in Paris."


def test_extract_claim_citations_drops_out_of_range_numbers():
    chunks = [_chunk("Only one chunk here.")]
    answer = "This claim cites a chunk that does not exist [7]."

    pairs = extract_claim_citations(answer, chunks)

    assert pairs == []


def test_extract_claim_citations_sentence_with_no_citation_is_skipped():
    chunks = [_chunk("Some source text.")]
    answer = "This sentence has no citation at all."

    assert extract_claim_citations(answer, chunks) == []


# --- verify_citation(s): real model, judging actual support ---

@requires_groq
def test_verify_citations_flags_a_genuinely_unsupported_claim():
    chunks = [_chunk("The company was founded in 2010 and is headquartered in Berlin.")]
    # deliberately claims something the passage does not say
    answer = "The company has over 10,000 employees worldwide [1]."

    results = verify_citations(answer, chunks)

    assert len(results) == 1
    assert results[0].supported is False


@requires_groq
def test_verify_citations_confirms_a_genuinely_supported_claim():
    chunks = [_chunk("The company was founded in 2010 and is headquartered in Berlin.")]
    answer = "The company was founded in 2010 [1]."

    results = verify_citations(answer, chunks)

    assert len(results) == 1
    assert results[0].supported is True
