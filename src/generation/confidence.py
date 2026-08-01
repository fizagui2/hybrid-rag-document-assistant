"""Answer confidence scoring: three sub-scores combined into one composite.

- retrieval_confidence: did we find at least one strongly relevant chunk?
  The cross-encoder's rerank_score is a raw logit (any real number, not
  0-1), so it's passed through a sigmoid to get something interpretable as
  a per-chunk relevance probability, then the MAX across context chunks is
  used (not the average) — retrieving several weaker supporting chunks
  alongside one excellent match is normal and expected, and averaging would
  wrongly punish that instead of rewarding having found the right answer.
- citation_coverage: what fraction of the answer's sentences have at least
  one citation that verify_citations() actually confirmed as supported?
  Penalizes both an uncited claim and a claim whose citation didn't hold up.
- completeness: does the answer address the whole question, not just part
  of it? A judgment call, so it's a third LLM-as-judge call.
"""
import math
import re
from dataclasses import dataclass

from ..retrieval.reranker import RerankedHit
from ..text import split_sentences
from .citations import VerifiedCitation
from .llm import chat

DEFAULT_WEIGHTS = (1 / 3, 1 / 3, 1 / 3)  # (retrieval, citation, completeness)

COMPLETENESS_SYSTEM_PROMPT = """You judge whether an answer fully addresses all parts of a question.

Respond with exactly one line: a single number from 0 to 100, where 100 means every part of the question was addressed and 0 means none of it was. No other text."""


@dataclass
class ConfidenceScore:
    retrieval_confidence: float
    citation_coverage: float
    completeness: float
    composite: float


def retrieval_confidence(chunks: list[RerankedHit]) -> float:
    if not chunks:
        return 0.0
    sigmoid_scores = [1 / (1 + math.exp(-chunk.rerank_score)) for chunk in chunks]
    return max(sigmoid_scores)


def citation_coverage(answer_text: str, verified: list[VerifiedCitation]) -> float:
    sentences = split_sentences(answer_text)
    if not sentences:
        return 0.0
    supported_claims = {v.claim for v in verified if v.supported}
    return len(supported_claims) / len(sentences)


def score_completeness(question: str, answer_text: str) -> float:
    messages = [
        {"role": "system", "content": COMPLETENESS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer_text}"},
    ]
    response = chat(messages, temperature=0.0)
    match = re.search(r"\d+(\.\d+)?", response)
    if not match:
        return 0.0
    return min(max(float(match.group()) / 100, 0.0), 1.0)


def score_confidence(
    question: str,
    answer_text: str,
    chunks: list[RerankedHit],
    verified_citations: list[VerifiedCitation],
    weights: tuple[float, float, float] = DEFAULT_WEIGHTS,
) -> ConfidenceScore:
    retrieval = retrieval_confidence(chunks)
    coverage = citation_coverage(answer_text, verified_citations)
    completeness = score_completeness(question, answer_text)

    w_retrieval, w_coverage, w_completeness = weights
    composite = w_retrieval * retrieval + w_coverage * coverage + w_completeness * completeness

    return ConfidenceScore(
        retrieval_confidence=retrieval,
        citation_coverage=coverage,
        completeness=completeness,
        composite=composite,
    )
