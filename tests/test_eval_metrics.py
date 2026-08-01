from src.eval.metrics import (
    EvalResult,
    citation_accuracy,
    retrieval_relevance,
    run_eval_case,
    score_correctness,
    score_faithfulness,
    summarize_results,
)
from src.generation.citations import VerifiedCitation
from src.ingestion.models import Chunk
from src.retrieval.index import HybridIndex
from src.retrieval.reranker import RerankedHit
from tests.conftest import requires_groq


def _hit(text, source_path="doc.md"):
    return RerankedHit(
        chunk_id="x", text=text, metadata={"source_path": source_path}, rerank_score=1.0, fused_score=1.0
    )


def _verified(supported):
    return VerifiedCitation(claim="claim", citation_number=1, supported=supported, reasoning="")


def _result(**overrides):
    defaults = dict(
        case_id="x", category="straightforward", question="q", answer_text="a", low_confidence=False,
        correctness=1.0, faithfulness=1.0, retrieval_relevance=1.0, citation_accuracy=1.0,
    )
    defaults.update(overrides)
    return EvalResult(**defaults)


# --- citation_accuracy: pure logic ---

def test_citation_accuracy_all_supported():
    assert citation_accuracy([_verified(True), _verified(True)]) == 1.0


def test_citation_accuracy_mixed():
    assert citation_accuracy([_verified(True), _verified(False)]) == 0.5


def test_citation_accuracy_no_citations_is_none():
    assert citation_accuracy([]) is None


# --- retrieval_relevance: pure logic ---

def test_retrieval_relevance_all_expected_docs_found():
    sources = [_hit("t1", "a.md"), _hit("t2", "b.md")]
    assert retrieval_relevance(sources, ["a.md", "b.md"]) == 1.0


def test_retrieval_relevance_partial_multi_hop_recall():
    # multi-hop expects two docs but only one was actually retrieved
    sources = [_hit("t1", "a.md")]
    assert retrieval_relevance(sources, ["a.md", "b.md"]) == 0.5


def test_retrieval_relevance_none_found():
    sources = [_hit("t1", "unrelated.md")]
    assert retrieval_relevance(sources, ["a.md"]) == 0.0


def test_retrieval_relevance_no_expected_docs_returns_none():
    assert retrieval_relevance([], []) is None


# --- summarize_results: pure logic ---

def test_summarize_results_computes_overall_and_per_category_means():
    results = [
        _result(case_id="s1", category="straightforward", correctness=1.0),
        _result(case_id="s2", category="straightforward", correctness=0.5),
        _result(case_id="m1", category="multi_hop", correctness=0.8),
    ]

    summary = summarize_results(results)

    assert summary["overall"]["count"] == 3
    assert summary["by_category"]["straightforward"]["count"] == 2
    assert summary["by_category"]["straightforward"]["correctness"] == 0.75
    assert summary["by_category"]["multi_hop"]["correctness"] == 0.8


def test_summarize_results_ignores_none_values_in_mean():
    results = [
        _result(case_id="n1", category="no_answer", faithfulness=None),
        _result(case_id="n2", category="no_answer", faithfulness=0.6),
    ]

    summary = summarize_results(results)

    # mean of just the one non-None value, not skewed by treating None as 0
    assert summary["by_category"]["no_answer"]["faithfulness"] == 0.6


# --- score_correctness / score_faithfulness / run_eval_case: real judge model ---

@requires_groq
def test_score_correctness_high_for_semantically_matching_answer():
    score = score_correctness(
        "How many vacation days per year?",
        "15 vacation days per year.",
        "Employees get fifteen days of vacation annually.",
    )
    assert score > 0.7


@requires_groq
def test_score_correctness_low_for_contradicting_answer():
    score = score_correctness(
        "How many vacation days per year?",
        "15 vacation days per year.",
        "Employees get 25 vacation days per year.",
    )
    assert score < 0.5


@requires_groq
def test_score_correctness_high_when_both_decline():
    score = score_correctness(
        "What is the referral bonus policy?",
        "The documents do not contain any information about employee referral bonuses.",
        "I don't have confidently relevant information to answer this question.",
    )
    assert score > 0.7


@requires_groq
def test_score_faithfulness_penalizes_an_unsupported_claim():
    chunks = [_hit("The company was founded in 2010 and is headquartered in Berlin.")]
    # second sentence is not grounded in the chunk at all
    answer = "The company was founded in 2010. The company has over 10,000 employees worldwide."

    score = score_faithfulness(answer, chunks)

    assert score == 0.5


@requires_groq
def test_run_eval_case_end_to_end(tmp_path):
    index = HybridIndex(chroma_dir=tmp_path / "chroma", bm25_path=tmp_path / "bm25.pkl")
    index.add([Chunk(text="Employees may work remotely up to 3 days per week.", source_path="policy.md", chunk_index=0, strategy="structure")])

    case = {
        "id": "test-1",
        "question": "How many days can I work remotely?",
        "expected_answer": "3 days per week.",
        "category": "straightforward",
        "source_documents": ["policy.md"],
    }

    result = run_eval_case(index, case)

    assert result.case_id == "test-1"
    assert result.low_confidence is False
    assert result.correctness > 0.7
    assert result.retrieval_relevance == 1.0
    assert result.citation_accuracy == 1.0
