import math

from src.generation.citations import VerifiedCitation
from src.generation.confidence import citation_coverage, retrieval_confidence, score_completeness, score_confidence
from src.retrieval.reranker import RerankedHit
from tests.conftest import requires_groq


def _hit(chunk_id, rerank_score):
    return RerankedHit(chunk_id=chunk_id, text="text", metadata={}, rerank_score=rerank_score, fused_score=0.0)


def _verified(claim, supported):
    return VerifiedCitation(claim=claim, citation_number=1, supported=supported, reasoning="")


# --- retrieval_confidence: pure math, no model needed ---

def test_retrieval_confidence_of_score_zero_is_half():
    # sigmoid(0) == 0.5 exactly — a neutral logit means neutral confidence
    assert math.isclose(retrieval_confidence([_hit("a", 0.0)]), 0.5, rel_tol=1e-9)


def test_retrieval_confidence_high_score_is_near_one():
    assert retrieval_confidence([_hit("a", 10.0)]) > 0.99


def test_retrieval_confidence_very_negative_score_is_near_zero():
    assert retrieval_confidence([_hit("a", -10.0)]) < 0.01


def test_retrieval_confidence_uses_the_best_chunk_not_the_average():
    # One excellent match plus several weak/irrelevant supporting chunks is
    # normal (final_k commonly retrieves more than one chunk of context) and
    # should NOT be penalized just because not every chunk is a great match
    # — what matters is whether at least one genuinely relevant chunk exists.
    mixed = retrieval_confidence([_hit("a", 10.0), _hit("b", -10.0), _hit("c", -5.0), _hit("d", -3.0)])
    high = retrieval_confidence([_hit("a", 10.0)])
    assert math.isclose(mixed, high, rel_tol=1e-9)


def test_retrieval_confidence_empty_is_zero():
    assert retrieval_confidence([]) == 0.0


# --- citation_coverage: pure logic, no model needed ---

def test_citation_coverage_all_sentences_supported():
    answer = "Paris is the capital of France. The Eiffel Tower is in Paris."
    verified = [_verified("Paris is the capital of France.", True), _verified("The Eiffel Tower is in Paris.", True)]

    assert citation_coverage(answer, verified) == 1.0


def test_citation_coverage_partial_when_some_unsupported():
    answer = "Paris is the capital of France. The Eiffel Tower is in Paris."
    verified = [_verified("Paris is the capital of France.", True), _verified("The Eiffel Tower is in Paris.", False)]

    assert citation_coverage(answer, verified) == 0.5


def test_citation_coverage_penalizes_uncited_sentences():
    # second sentence has no citation at all, so it never appears in `verified`
    answer = "Paris is the capital of France. This sentence cites nothing."
    verified = [_verified("Paris is the capital of France.", True)]

    assert citation_coverage(answer, verified) == 0.5


def test_citation_coverage_empty_answer_is_zero():
    assert citation_coverage("", []) == 0.0


# --- score_completeness / score_confidence: real judge model ---

@requires_groq
def test_score_completeness_high_for_fully_addressed_question():
    question = "What is the capital of France?"
    answer = "The capital of France is Paris."

    score = score_completeness(question, answer)

    assert score > 0.7


@requires_groq
def test_score_completeness_low_for_partially_addressed_question():
    question = "What is the capital of France, and what is its population?"
    answer = "The capital of France is Paris."  # never answers the population half

    score = score_completeness(question, answer)

    assert score < 0.7


@requires_groq
def test_score_confidence_combines_all_three_dimensions():
    chunks = [_hit("a", 5.0)]
    answer_text = "Paris is the capital of France."
    verified = [_verified("Paris is the capital of France.", True)]

    result = score_confidence("What is the capital of France?", answer_text, chunks, verified)

    assert 0.0 <= result.retrieval_confidence <= 1.0
    assert result.citation_coverage == 1.0
    assert 0.0 <= result.completeness <= 1.0
    assert 0.0 <= result.composite <= 1.0
