"""Automated evaluation: run a golden Q&A case through the full pipeline and
score it against a known-correct reference on four dimensions.

- correctness: LLM-as-judge comparing the generated answer to the golden
  case's expected_answer. Explicitly told that a graceful decline counts as
  correct when the reference says the information isn't available —
  otherwise every no_answer case would score "wrong" for correctly
  refusing to hallucinate.
- faithfulness: what fraction of the answer's claims are grounded in the
  retrieved context AT ALL (any chunk, not just whichever one got cited)?
  Reuses verify_citation() from citations.py — checking a claim against
  the full concatenated context is the exact same judgment task as
  checking it against one specific cited chunk, just different source
  text. None when the answer is a low-confidence refusal (nothing to
  fact-check).
- retrieval_relevance: for cases with known expected source documents
  (straightforward, multi_hop), the fraction of those documents that
  actually appear among the retrieved chunks — for multi_hop this directly
  tests whether BOTH expected documents were found, not just one. For
  no_answer cases (no correct source exists by definition), scored as 1.0
  if the system correctly declined and 0.0 if it forced an answer anyway.
  None for ambiguous cases, where declining isn't necessarily wrong the
  way it is for no_answer, so there's no single correct behavior to score.
- citation_accuracy: fraction of the answer's own citations that were
  verified as actually supported — reuses result.citations directly from
  answer_question(), no extra LLM calls needed.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..generation.answer import Answer, answer_question
from ..generation.citations import ClaimCitation, VerifiedCitation, verify_citation
from ..generation.llm import chat
from ..retrieval.index import HybridIndex
from ..text import split_sentences

CORRECTNESS_SYSTEM_PROMPT = """You judge whether a system's answer is correct compared to a reference answer, for a question-answering system that may also decline to answer when it lacks confident information.

Consider the system's answer CORRECT if it conveys the same essential information as the reference answer, even with different wording. If the reference answer says the information is not available (or that the question is ambiguous), and the system's answer also declines to answer or says it lacks enough information, that counts as CORRECT.

Respond with exactly two lines:
Line 1: a single number from 0 to 100, where 100 means fully correct and 0 means completely wrong or contradicts the reference
Line 2: a one-sentence reason"""


@dataclass
class EvalResult:
    case_id: str
    category: str
    question: str
    answer_text: str
    low_confidence: bool
    correctness: float
    faithfulness: Optional[float]
    retrieval_relevance: Optional[float]
    citation_accuracy: Optional[float]


def score_correctness(question: str, expected_answer: str, actual_answer: str) -> float:
    messages = [
        {"role": "system", "content": CORRECTNESS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\n\nReference answer: {expected_answer}\n\nSystem's answer: {actual_answer}",
        },
    ]
    response = chat(messages, temperature=0.0)
    match = re.search(r"\d+(\.\d+)?", response)
    if not match:
        return 0.0
    return min(max(float(match.group()) / 100, 0.0), 1.0)


def score_faithfulness(answer_text: str, chunks: list) -> float:
    sentences = split_sentences(answer_text)
    if not sentences:
        return 0.0
    full_context = "\n\n".join(chunk.text for chunk in chunks)
    verified = [
        verify_citation(ClaimCitation(claim=sentence, citation_number=0, chunk_text=full_context))
        for sentence in sentences
    ]
    return sum(1 for v in verified if v.supported) / len(verified)


def citation_accuracy(verified_citations: list[VerifiedCitation]) -> Optional[float]:
    if not verified_citations:
        return None
    return sum(1 for v in verified_citations if v.supported) / len(verified_citations)


def retrieval_relevance(sources: list, expected_documents: list[str]) -> Optional[float]:
    """Deliberately independent of whether the answer was actually generated
    (low_confidence) — this measures retrieval quality on its own, so a case
    where the right document WAS retrieved but the confidence gate blocked
    generation can be told apart from a case where retrieval itself failed.
    Both are real failure modes, but they point at different parts of the
    pipeline to fix.
    """
    if not expected_documents:
        return None
    retrieved_docs = {Path(s.metadata.get("source_path", "")).name for s in sources}
    found = set(expected_documents) & retrieved_docs
    return len(found) / len(expected_documents)


def run_eval_case(index: HybridIndex, case: dict) -> EvalResult:
    """Run one golden-dataset case through the real pipeline and score it."""
    result: Answer = answer_question(index, case["question"], verify=True)

    correctness = score_correctness(case["question"], case["expected_answer"], result.text)

    if result.low_confidence:
        faithfulness = None
        accuracy = None
    else:
        faithfulness = score_faithfulness(result.text, result.sources)
        accuracy = citation_accuracy(result.citations)

    expected_documents = case["source_documents"]
    if expected_documents:
        relevance = retrieval_relevance(result.sources, expected_documents)
    elif case["category"] == "no_answer":
        relevance = 1.0 if result.low_confidence else 0.0
    else:
        relevance = None

    return EvalResult(
        case_id=case["id"],
        category=case["category"],
        question=case["question"],
        answer_text=result.text,
        low_confidence=result.low_confidence,
        correctness=correctness,
        faithfulness=faithfulness,
        retrieval_relevance=relevance,
        citation_accuracy=accuracy,
    )


def run_eval_suite(index: HybridIndex, cases: list[dict]) -> list[EvalResult]:
    return [run_eval_case(index, case) for case in cases]


def _mean(values: list[float]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def summarize_results(results: list[EvalResult]) -> dict:
    """Mean of each metric, overall and broken down per category."""
    def summarize(subset: list[EvalResult]) -> dict:
        return {
            "count": len(subset),
            "correctness": _mean([r.correctness for r in subset]),
            "faithfulness": _mean([r.faithfulness for r in subset]),
            "retrieval_relevance": _mean([r.retrieval_relevance for r in subset]),
            "citation_accuracy": _mean([r.citation_accuracy for r in subset]),
        }

    categories = sorted({r.category for r in results})
    return {
        "overall": summarize(results),
        "by_category": {category: summarize([r for r in results if r.category == category]) for category in categories},
    }
