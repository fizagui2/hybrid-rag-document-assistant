"""Ties retrieval + prompting + generation + verification into one call:
ask a question, get back a grounded answer with citations, verification,
and a confidence score, using the top reranked chunks from the whole
Phase 2 pipeline as context.

If the retrieved chunks aren't confidently relevant to begin with, the LLM
is never asked to generate an answer from them at all — see
_build_low_confidence_answer. Forcing a generation call on weak context is
exactly how RAG systems end up hallucinating a confident-sounding wrong
answer; refusing gracefully, with pointers to what might be worth checking
manually, is more useful and costs zero LLM calls.
"""
from dataclasses import dataclass, field
from typing import Optional

from ..retrieval.index import HybridIndex
from ..retrieval.reranker import RerankedHit, retrieve
from .citations import VerifiedCitation, verify_citations
from .confidence import ConfidenceScore, DEFAULT_WEIGHTS, retrieval_confidence, score_confidence
from .llm import chat
from .prompts import build_messages

NO_CONTEXT_ANSWER = "I don't have any indexed documents relevant to this question."

# retrieval_confidence is a sigmoid of the cross-encoder's raw score, so 0.5
# is exactly where that raw score crosses zero — the cross-encoder's own
# boundary between "leans relevant" and "leans irrelevant," not a number
# picked by guesswork.
LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class Answer:
    question: str
    text: str
    sources: list[RerankedHit]
    citations: list[VerifiedCitation] = field(default_factory=list)
    confidence: Optional[ConfidenceScore] = None
    low_confidence: bool = False


def _build_low_confidence_answer(chunks: list[RerankedHit]) -> str:
    candidates = []
    for chunk in chunks:
        location = chunk.metadata.get("source_path", "unknown")
        if chunk.metadata.get("heading"):
            location += f" — {chunk.metadata['heading']}"
        candidates.append(f"- {location}")

    return (
        "I don't have confidently relevant information to answer this question. "
        "None of the closest matches in the indexed documents were strongly relevant, "
        "but these might be worth checking manually:\n" + "\n".join(candidates)
    )


def _low_confidence_score(retrieval_score: float, weights: tuple[float, float, float] = DEFAULT_WEIGHTS) -> ConfidenceScore:
    """No answer was generated, so citation_coverage and completeness are
    both trivially 0 (there's no cited content, and nothing addressed the
    question) — composite uses the same weighted formula as score_confidence
    for consistency, it just has two known-zero terms.
    """
    w_retrieval, _, _ = weights
    return ConfidenceScore(
        retrieval_confidence=retrieval_score,
        citation_coverage=0.0,
        completeness=0.0,
        composite=w_retrieval * retrieval_score,
    )


def answer_question(
    index: HybridIndex,
    question: str,
    final_k: int = 5,
    verify: bool = True,
    min_confidence: float = LOW_CONFIDENCE_THRESHOLD,
) -> Answer:
    """verify=True (the default) runs citation verification + confidence
    scoring, which costs extra LLM calls (one per cited sentence, plus one
    for completeness) on top of generation itself. Set verify=False for a
    faster, bare answer — e.g. when running many questions in bulk later
    during evaluation, where verifying every single one may not be needed.
    """
    chunks = retrieve(index, question, final_k=final_k)
    if not chunks:
        return Answer(question=question, text=NO_CONTEXT_ANSWER, sources=[], low_confidence=True)

    retrieval_score = retrieval_confidence(chunks)
    if retrieval_score < min_confidence:
        text = _build_low_confidence_answer(chunks)
        return Answer(
            question=question,
            text=text,
            sources=chunks,
            confidence=_low_confidence_score(retrieval_score),
            low_confidence=True,
        )

    messages = build_messages(question, chunks)
    text = chat(messages)

    if not verify:
        return Answer(question=question, text=text, sources=chunks)

    verified = verify_citations(text, chunks)
    confidence = score_confidence(question, text, chunks, verified)
    return Answer(question=question, text=text, sources=chunks, citations=verified, confidence=confidence)
